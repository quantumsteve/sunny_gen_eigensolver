// Inspector for the .bin reference produced by lapack_reference.
// Verifies the magic/version, prints batch sizes, and reports basic statistics
// for sanity checking: eigenvalue ranges and Hermiticity-floor diagnostics.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic   = 0x47454947;
constexpr std::int32_t kVersion = 1;

template <typename T>
bool read_pod(std::ifstream& f, T& v) {
    return static_cast<bool>(
        f.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dump_pair path.bin\n");
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 3;
    }

    std::int32_t magic, version, N, B, seed, reserved;
    if (!read_pod(f, magic) || !read_pod(f, version) || !read_pod(f, N) ||
        !read_pod(f, B)     || !read_pod(f, seed)    || !read_pod(f, reserved)) {
        std::fprintf(stderr, "header read failed\n");
        return 4;
    }
    if (magic != kMagic) {
        std::fprintf(stderr, "bad magic 0x%08x (expected 0x%08x)\n", magic, kMagic);
        return 5;
    }
    if (version != kVersion) {
        std::fprintf(stderr, "unsupported version %d (expected %d)\n", version, kVersion);
        return 6;
    }

    const std::size_t mat_elems = static_cast<std::size_t>(N) * N;
    std::vector<cdouble> H(mat_elems * B);
    std::vector<cdouble> S(mat_elems * B);
    std::vector<double>  W(static_cast<std::size_t>(N) * B);
    std::vector<cdouble> U(mat_elems * B);

    auto read_block = [&](void* p, std::size_t bytes, const char* what) {
        if (!f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(bytes))) {
            std::fprintf(stderr, "short read in %s\n", what);
            std::exit(7);
        }
    };
    read_block(H.data(), sizeof(cdouble) * H.size(), "H");
    read_block(S.data(), sizeof(cdouble) * S.size(), "S");
    read_block(W.data(), sizeof(double)  * W.size(), "W");
    read_block(U.data(), sizeof(cdouble) * U.size(), "U");

    std::printf("file:    %s\n", argv[1]);
    std::printf("magic:   0x%08x\n", magic);
    std::printf("version: %d\n",     version);
    std::printf("N:       %d\n",     N);
    std::printf("B:       %d\n",     B);
    std::printf("seed:    %d\n",     seed);

    // Per-matrix eigenvalue range, plus check that eigenvalues come out sorted
    // and that LAPACK's W is real (it always is — diagnostic confirms our format).
    double w_imag_floor = 0.0;  // |Im W|, but W is real-typed by LAPACK; this is 0 by construction.

    // Hermiticity floor on input H: max |H_ij - conj(H_ji)|.
    // Useful as a sanity check that we wrote the matrices we think we wrote.
    double herm_floor = 0.0;

    for (int b = 0; b < B; ++b) {
        const cdouble* Hb = &H[mat_elems * b];
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                cdouble hij = Hb[static_cast<std::size_t>(j) * N + i];
                cdouble hji = Hb[static_cast<std::size_t>(i) * N + j];
                herm_floor = std::max(herm_floor, std::abs(hij - std::conj(hji)));
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        const double* Wb = &W[static_cast<std::size_t>(N) * b];
        double mn = Wb[0], mx = Wb[0];
        bool sorted = true;
        for (int i = 0; i < N; ++i) {
            mn = std::min(mn, Wb[i]);
            mx = std::max(mx, Wb[i]);
            if (i > 0 && Wb[i] < Wb[i - 1]) sorted = false;
        }
        std::printf("batch %d: eigenvalues [%.6e, %.6e]   sorted_ascending=%s\n",
                    b, mn, mx, sorted ? "yes" : "NO");
    }

    std::printf("max |Im W|:                      %.3e (LAPACK W is real-typed; expect 0)\n", w_imag_floor);
    std::printf("max Hermiticity floor on H:      %.3e (expect ~0)\n", herm_floor);
    return 0;
}
