#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_gpu.h>
#include <torch/python.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "apis/mega_dcu.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

namespace deep_gemm::runtime {

static int g_num_sms = 0;
static int g_tc_util = 100;
static bool g_pdl = false;
static int g_mk_alignment = 1;

#define DG_HIP_CHECK(expr) do { \
    const hipError_t status = (expr); \
    if (status != hipSuccess) { \
        throw std::runtime_error(std::string(#expr) + " failed: " + hipGetErrorString(status)); \
    } \
} while (0)

#define DG_HSA_CHECK(expr) do { \
    const hsa_status_t status = (expr); \
    if (status != HSA_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string(#expr) + " failed with HSA status " + std::to_string(static_cast<int>(status))); \
    } \
} while (0)

struct DevAgentInfo {
    int64_t bus_id = 0;
    hsa_agent_t hsa_agent{};
};

static void ensure_hsa_initialized() {
    static std::once_flag once;
    std::call_once(once, []() {
        DG_HSA_CHECK(hsa_init());
    });
}

static int64_t align_fabric_bytes(const int64_t num_bytes) {
    constexpr int64_t kFabricAlignmentBytes = 2 * 1024 * 1024;
    return (num_bytes + kFabricAlignmentBytes - 1) & ~(kFabricAlignmentBytes - 1);
}

static bool bus_id_to_int64(const char* bus_id, int64_t* id) {
    char hex_str[17];
    int hex_offset = 0;
    for (int i = 0; hex_offset < static_cast<int>(sizeof(hex_str)) - 1; ++i) {
        const char c = bus_id[i];
        if (c == '.' || c == ':')
            continue;
        if ((c >= '0' && c <= '9') ||
            (c >= 'A' && c <= 'F') ||
            (c >= 'a' && c <= 'f')) {
            hex_str[hex_offset++] = c;
        } else {
            break;
        }
    }
    hex_str[hex_offset] = '\0';
    *id = std::strtoll(hex_str, nullptr, 16);
    return hex_offset > 0;
}

static hsa_status_t iterate_agent_callback(hsa_agent_t agent, void* data) {
    auto* agent_info = static_cast<DevAgentInfo*>(data);
    uint32_t hsa_pci_bdf_id = 0;
    uint32_t hsa_pci_domain_id = 0;

    hsa_status_t status = hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
        &hsa_pci_bdf_id);
    if (status != HSA_STATUS_SUCCESS)
        return status;
    status = hsa_agent_get_info(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
        &hsa_pci_domain_id);
    if (status != HSA_STATUS_SUCCESS)
        return status;

    const uint32_t hsa_bus = (hsa_pci_bdf_id >> 8) & 0xff;
    const uint32_t hsa_device = (hsa_pci_bdf_id >> 3) & 0x1f;
    const uint32_t hsa_function = hsa_pci_bdf_id & 0x07;

    const uint32_t dev_domain_id = static_cast<uint32_t>(agent_info->bus_id >> 20);
    const uint32_t dev_bus = static_cast<uint32_t>((agent_info->bus_id >> 12) & 0xff);
    const uint32_t dev_device = static_cast<uint32_t>((agent_info->bus_id >> 4) & 0x1f);
    const uint32_t dev_function = static_cast<uint32_t>(agent_info->bus_id & 0x07);

    if (hsa_pci_domain_id == dev_domain_id && hsa_bus == dev_bus &&
        hsa_device == dev_device && hsa_function == dev_function) {
        agent_info->hsa_agent = agent;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_agent_t get_current_device_agent() {
    ensure_hsa_initialized();
    int device = 0;
    DG_HIP_CHECK(hipGetDevice(&device));
    char bus_id_str[] = "00000000:00:00.0";
    DG_HIP_CHECK(hipDeviceGetPCIBusId(bus_id_str, sizeof(bus_id_str), device));

    DevAgentInfo agent_info{};
    TORCH_CHECK(bus_id_to_int64(bus_id_str, &agent_info.bus_id),
                "failed to parse HIP PCI bus id for HSA agent lookup");
    DG_HSA_CHECK(hsa_iterate_agents(iterate_agent_callback, &agent_info));
    TORCH_CHECK(agent_info.hsa_agent.handle != 0,
                "failed to find matching HSA agent for current HIP device");
    return agent_info.hsa_agent;
}

static pybind11::bytes make_fabric_handle_bytes(void* ptr, const int64_t num_bytes) {
    TORCH_CHECK(num_bytes > 0, "fabric buffer size must be positive");
    ensure_hsa_initialized();
    hsa_ext_rpc_memory_t handle{};
    DG_HSA_CHECK(hsa_ext_rpc_memory_create(
        ptr, static_cast<size_t>(num_bytes), &handle));
    return pybind11::bytes(
        reinterpret_cast<const char*>(&handle), sizeof(hsa_ext_rpc_memory_t));
}

static pybind11::bytes make_hip_ipc_handle_bytes(void* ptr) {
    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
    return pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE);
}

static pybind11::bytes get_hip_ipc_handle(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "HIP IPC handle requires a CUDA/HIP tensor");
    return make_hip_ipc_handle_bytes(reinterpret_cast<void*>(tensor.data_ptr()));
}

static pybind11::tuple allocate_hip_ipc_buffer(const int64_t& num_bytes) {
    TORCH_CHECK(num_bytes > 0, "HIP IPC buffer size must be positive");
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(num_bytes), hipDeviceMallocFinegrained));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(num_bytes)));

    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
    auto tensor = torch::from_blob(
        ptr,
        {num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    return pybind11::make_tuple(
        tensor,
        reinterpret_cast<int64_t>(ptr),
        pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE));
}

static pybind11::tuple allocate_hip_fabric_buffer(const int64_t& num_bytes) {
    TORCH_CHECK(num_bytes > 0, "HIP fabric buffer size must be positive");
    const int64_t alloc_bytes = align_fabric_bytes(num_bytes);
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(alloc_bytes), hipDeviceMallocFinegrained));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(alloc_bytes)));

    auto tensor = torch::from_blob(
        ptr,
        {num_bytes},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    return pybind11::make_tuple(
        tensor,
        reinterpret_cast<int64_t>(ptr),
        make_fabric_handle_bytes(ptr, alloc_bytes));
}

static pybind11::tuple allocate_hip_fabric_signal_buffer(const int64_t& num_bytes) {
    TORCH_CHECK(num_bytes > 0, "HIP fabric signal buffer size must be positive");
    const int64_t alloc_bytes = align_fabric_bytes(num_bytes);
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(alloc_bytes), hipDeviceMallocFinegrained));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(alloc_bytes)));

    return pybind11::make_tuple(
        reinterpret_cast<int64_t>(ptr),
        make_fabric_handle_bytes(ptr, alloc_bytes));
}

static pybind11::tuple allocate_hip_ipc_signal_buffer(const int64_t& num_bytes) {
    TORCH_CHECK(num_bytes > 0, "HIP IPC signal buffer size must be positive");
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(num_bytes), hipDeviceMallocUncached));
    DG_HIP_CHECK(hipMemset(ptr, 0, static_cast<size_t>(num_bytes)));

    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, ptr));
    return pybind11::make_tuple(
        reinterpret_cast<int64_t>(ptr),
        pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE));
}

static std::vector<int64_t> open_hip_fabric_handles(const std::vector<pybind11::bytes>& handles,
                                                   const int& local_rank) {
    TORCH_CHECK(local_rank >= 0 && local_rank < static_cast<int>(handles.size()),
                "local_rank is out of bounds for HIP fabric handles");
    const hsa_agent_t agent = get_current_device_agent();
    std::vector<int64_t> ptrs(handles.size(), 0);
    for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
        if (i == local_rank)
            continue;

        const std::string handle_str = handles[i];
        TORCH_CHECK(handle_str.size() == sizeof(hsa_ext_rpc_memory_t),
                    "invalid HIP fabric handle size");

        hsa_ext_rpc_memory_t handle{};
        std::memcpy(&handle, handle_str.data(), sizeof(hsa_ext_rpc_memory_t));

        void* ptr = nullptr;
        DG_HSA_CHECK(hsa_ext_rpc_memory_attach(&handle, 1, &agent, &ptr));
        ptrs[i] = reinterpret_cast<int64_t>(ptr);
    }
    return ptrs;
}

static std::vector<int64_t> open_hip_ipc_handles(const std::vector<pybind11::bytes>& handles,
                                                const int& local_rank) {
    TORCH_CHECK(local_rank >= 0 && local_rank < static_cast<int>(handles.size()),
                "local_rank is out of bounds for HIP IPC handles");
    std::vector<int64_t> ptrs(handles.size(), 0);
    for (int i = 0; i < static_cast<int>(handles.size()); ++i) {
        if (i == local_rank)
            continue;

        const std::string handle_str = handles[i];
        TORCH_CHECK(handle_str.size() == HIP_IPC_HANDLE_SIZE, "invalid HIP IPC handle size");

        hipIpcMemHandle_t handle{};
        std::memcpy(handle.reserved, handle_str.data(), HIP_IPC_HANDLE_SIZE);

        void* ptr = nullptr;
        DG_HIP_CHECK(hipIpcOpenMemHandle(&ptr, handle, hipIpcMemLazyEnablePeerAccess));
        ptrs[i] = reinterpret_cast<int64_t>(ptr);
    }
    return ptrs;
}

static void close_hip_fabric_handles(const std::vector<int64_t>& ptrs) {
    for (const auto ptr_value: ptrs) {
        if (ptr_value != 0)
            DG_HSA_CHECK(hsa_ext_rpc_memory_detach(reinterpret_cast<void*>(ptr_value)));
    }
}

static void close_hip_ipc_handles(const std::vector<int64_t>& ptrs) {
    for (const auto ptr_value: ptrs) {
        if (ptr_value != 0)
            DG_HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(ptr_value)));
    }
}

static void free_hip_ipc_signal_buffer(const int64_t& ptr_value) {
    if (ptr_value != 0)
        DG_HIP_CHECK(hipFree(reinterpret_cast<void*>(ptr_value)));
}

static void register_apis(pybind11::module_& m) {
    m.def("set_num_sms", [](const int& new_num_sms) { g_num_sms = new_num_sms; });
    m.def("get_num_sms", []() { return g_num_sms; });
    m.def("set_tc_util", [](const int& new_tc_util) { g_tc_util = new_tc_util; });
    m.def("get_tc_util", []() { return g_tc_util; });
    m.def("set_pdl", [](const bool& new_enable_pdl) { g_pdl = new_enable_pdl; });
    m.def("get_pdl", []() { return g_pdl; });
    m.def("set_ignore_compile_dims", [](const bool&) {});
    m.def("set_block_size_multiple_of", [](const pybind11::object&) {});
    m.def("init", [](const std::string&, const std::string&) {});

    m.def("set_mk_alignment_for_contiguous_layout", [](const int& new_value) {
        g_mk_alignment = new_value;
    });
    m.def("get_mk_alignment_for_contiguous_layout", []() {
        return g_mk_alignment;
    });
    m.def("get_theoretical_mk_alignment_for_contiguous_layout", [](const std::optional<int>&) {
        return 1;
    }, pybind11::arg("expected_m") = std::nullopt);

    m.def("get_hip_ipc_handle", &get_hip_ipc_handle);
    m.def("allocate_hip_ipc_buffer", &allocate_hip_ipc_buffer);
    m.def("allocate_hip_ipc_signal_buffer", &allocate_hip_ipc_signal_buffer);
    m.def("allocate_hip_fabric_buffer", &allocate_hip_fabric_buffer);
    m.def("allocate_hip_fabric_signal_buffer", &allocate_hip_fabric_signal_buffer);
    m.def("open_hip_ipc_handles", &open_hip_ipc_handles);
    m.def("open_hip_fabric_handles", &open_hip_fabric_handles);
    m.def("close_hip_ipc_handles", &close_hip_ipc_handles);
    m.def("close_hip_fabric_handles", &close_hip_fabric_handles);
    m.def("free_hip_ipc_buffer", &free_hip_ipc_signal_buffer);
    m.def("free_hip_ipc_signal_buffer", &free_hip_ipc_signal_buffer);

    auto unsupported = []() {
        throw std::runtime_error("This CUDA/cuBLASLt API is not available in the megamoe HIP W8A8 extension");
    };
    m.def("cublaslt_gemm_nt", unsupported);
    m.def("cublaslt_gemm_nn", unsupported);
    m.def("cublaslt_gemm_tn", unsupported);
    m.def("cublaslt_gemm_tt", unsupported);
}

} // namespace deep_gemm::runtime

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "megamoe HIP W8A8 channelwise extension";
    deep_gemm::runtime::register_apis(m);
    deep_gemm::mega::register_apis(m);
}
