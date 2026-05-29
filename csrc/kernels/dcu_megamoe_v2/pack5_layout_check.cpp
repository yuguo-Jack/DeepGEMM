#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

constexpr int kK64 = 64;
constexpr int kN256 = 256;
constexpr int kK16 = 16;
constexpr int kN16 = 16;

int parse_int(const char* s) {
    return std::atoi(s);
}

int logical_to_physical_ni(int logical_ni) {
    return (logical_ni & 3) * 4 + (logical_ni >> 2);
}

int64_t pack5_flat_offset(int expert, int n, int k, int row, int col) {
    const int k_outer = k / kK64;
    const int n_outer = n / kN256;
    const int n16_outer = kN256 / kN16;
    const int k16_segment = kK64 / kK16;
    const int ko = col / kK64;
    const int no = row / kN256;
    const int ni16 = (row % kN256) / kN16;
    const int logical_ni = row % kN16;
    const int physical_ni = logical_to_physical_ni(logical_ni);
    const int ks = (col % kK64) / kK16;
    const int ki = col % kK16;
    return ((((((static_cast<int64_t>(expert) * k_outer + ko) * n_outer + no) *
               n16_outer + ni16) *
              k16_segment + ks) *
             kN16 + physical_ni) *
            kK16 + ki);
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --experts E --n N --k K --expert EID --row R --col C\n";
}

}  // namespace

int main(int argc, char** argv) {
    int experts = 0;
    int n = 0;
    int k = 0;
    int expert = 0;
    int row = 0;
    int col = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "--experts") {
            experts = parse_int(need_value("--experts"));
        } else if (key == "--n") {
            n = parse_int(need_value("--n"));
        } else if (key == "--k") {
            k = parse_int(need_value("--k"));
        } else if (key == "--expert") {
            expert = parse_int(need_value("--expert"));
        } else if (key == "--row") {
            row = parse_int(need_value("--row"));
        } else if (key == "--col") {
            col = parse_int(need_value("--col"));
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (experts <= 0 || n <= 0 || k <= 0 || n % kN256 != 0 || k % kK64 != 0 ||
        expert < 0 || expert >= experts || row < 0 || row >= n || col < 0 ||
        col >= k) {
        usage(argv[0]);
        return 2;
    }
    std::cout << pack5_flat_offset(expert, n, k, row, col) << "\n";
    return 0;
}
