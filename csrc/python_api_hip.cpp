#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <hip/hip_runtime.h>
#include <torch/python.h>

#include <cstring>
#include <stdexcept>
#include <string>
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

static pybind11::bytes get_hip_ipc_handle(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "HIP IPC handle requires a CUDA/HIP tensor");
    hipIpcMemHandle_t handle{};
    DG_HIP_CHECK(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(tensor.data_ptr())));
    return pybind11::bytes(handle.reserved, HIP_IPC_HANDLE_SIZE);
}

static pybind11::tuple allocate_hip_ipc_buffer(const int64_t& num_bytes) {
    TORCH_CHECK(num_bytes > 0, "HIP IPC buffer size must be positive");
    void* ptr = nullptr;
    DG_HIP_CHECK(hipExtMallocWithFlags(&ptr, static_cast<size_t>(num_bytes), hipDeviceMallocUncached));
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
    m.def("open_hip_ipc_handles", &open_hip_ipc_handles);
    m.def("close_hip_ipc_handles", &close_hip_ipc_handles);
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
