// LAPACK reference solver for the complex Hermitian-definite generalized
// eigenvalue problem H u = lambda S u.
//
// Generates a reproducible (H, S) batch from a 64-bit Mersenne Twister seeded
// with --seed, solves with LAPACKE_zhegvd, and writes the inputs together with
// the eigenvalues and eigenvectors to a binary file in the format declared in
// the file header below.

#include <lapacke.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using cdouble = std::complex<double>;

namespace {

constexpr std::int32_t kMagic   = 0x47454947;  // "GEIG"
constexpr std::int32_t kVersion = 1;

enum class SMode { Random, NearIdentity };

struct Args {
    int n             = 0;
    int batch         = 0;
    int seed          = 0;
    SMode s_mode      = SMode::Random;
    bool check_resid  = true;
    std::string out;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto need = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", flag);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (s == "--n")                   a.n     = std::stoi(need("--n"));
        else if (s == "--batch")               a.batch = std::stoi(need("--batch"));
        else if (s == "--seed")                a.seed  = std::stoi(need("--seed"));
        else if (s == "--out")                 a.out   = need("--out");
        else if (s == "--s-mode") {
            std::string v = need("--s-mode");
            if      (v == "random")        a.s_mode = SMode::Random;
            else if (v == "near-identity") a.s_mode = SMode::NearIdentity;
            else {
                std::fprintf(stderr, "unknown --s-mode value: %s "
                                     "(expected random|near-identity)\n", v.c_str());
                std::exit(2);
            }
        }
        else if (s == "--check-residuals")    a.check_resid = true;
        else if (s == "--no-check-residuals") a.check_resid = false;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", s.c_str());
            std::exit(2);
        }
    }
    if (a.n <= 0 || a.batch <= 0 || a.out.empty()) {
        std::fprintf(stderr,
            "usage: lapack_reference --n N --batch B --seed S --out path.bin\n"
            "                        [--s-mode random|near-identity]\n"
            "                        [--check-residuals|--no-check-residuals]\n");
        std::exit(2);
    }
    return a;
}

// Generate one Hermitian H of size N x N (column-major).
// H = A + A^H where A has entries N(0,1) + i*N(0,1). Eigenvalues are O(N) at
// large N (off-diagonal Wigner spread sqrt(N) plus 2*Re(A_ii) on the diagonal).
void make_hermitian(int n, std::mt19937_64& rng, cdouble* H) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            A[static_cast<std::size_t>(j) * n + i] = cdouble(g(rng), g(rng));
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            cdouble aij = A[static_cast<std::size_t>(j) * n + i];
            cdouble aji = A[static_cast<std::size_t>(i) * n + j];
            H[static_cast<std::size_t>(j) * n + i] = aij + std::conj(aji);
        }
    }
}

// Random S: Hermitian positive-definite, "wide spectrum".
// S = A * A^H + N * I. Eigenvalues span O(N) to O(N + 4N) ≈ O(5N).
void make_hpd_random(int n, std::mt19937_64& rng, cdouble* S) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> A(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            A[static_cast<std::size_t>(j) * n + i] = cdouble(g(rng), g(rng));
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            cdouble acc(0.0, 0.0);
            for (int k = 0; k < n; ++k) {
                cdouble aik = A[static_cast<std::size_t>(k) * n + i];
                cdouble ajk = A[static_cast<std::size_t>(k) * n + j];
                acc += aik * std::conj(ajk);
            }
            S[static_cast<std::size_t>(j) * n + i] = acc;
        }
    }
    for (int i = 0; i < n; ++i) {
        S[static_cast<std::size_t>(i) * n + i] += cdouble(static_cast<double>(n), 0.0);
    }
}

// Near-identity S: small Hermitian perturbation around identity.
// S = (1 + 0.5*N) * I + 0.1 * (B + B^H) for random B with unit-Gaussian entries.
// Models Sunny-like overlap matrices that are nearly diagonal-dominant.
void make_hpd_near_identity(int n, std::mt19937_64& rng, cdouble* S) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cdouble> B(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            B[static_cast<std::size_t>(j) * n + i] = cdouble(g(rng), g(rng));
        }
    }
    const double diag_shift = 1.0 + 0.5 * static_cast<double>(n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            cdouble bij = B[static_cast<std::size_t>(j) * n + i];
            cdouble bji = B[static_cast<std::size_t>(i) * n + j];
            cdouble val = 0.1 * (bij + std::conj(bji));
            if (i == j) val += cdouble(diag_shift, 0.0);
            S[static_cast<std::size_t>(j) * n + i] = val;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    const int N = args.n;
    const int B = args.batch;
    const std::size_t mat_elems = static_cast<std::size_t>(N) * N;

    std::vector<cdouble> H_all(mat_elems * B);
    std::vector<cdouble> S_all(mat_elems * B);
    std::vector<cdouble> U_all(mat_elems * B);     // eigenvectors
    std::vector<double>  W_all(static_cast<std::size_t>(N) * B);

    std::mt19937_64 rng(static_cast<std::uint64_t>(args.seed));

    // Generate inputs first (deterministic w.r.t. seed and s-mode).
    for (int b = 0; b < B; ++b) {
        make_hermitian(N, rng, &H_all[mat_elems * b]);
        if (args.s_mode == SMode::Random) {
            make_hpd_random(N, rng, &S_all[mat_elems * b]);
        } else {
            make_hpd_near_identity(N, rng, &S_all[mat_elems * b]);
        }
    }

    // Solve each pair. LAPACKE_zhegvd overwrites A with eigenvectors and B with
    // its Cholesky factor, so we work on copies.
    double evnorm_sq = 0.0;
    for (int b = 0; b < B; ++b) {
        std::vector<cdouble> H_work(mat_elems);
        std::vector<cdouble> S_work(mat_elems);
        std::memcpy(H_work.data(), &H_all[mat_elems * b], sizeof(cdouble) * mat_elems);
        std::memcpy(S_work.data(), &S_all[mat_elems * b], sizeof(cdouble) * mat_elems);

        lapack_int info = LAPACKE_zhegvd(
            LAPACK_COL_MAJOR,
            /*itype=*/1,         // A x = lambda B x
            /*jobz=*/'V',        // eigenvectors please
            /*uplo=*/'L',
            N,
            reinterpret_cast<lapack_complex_double*>(H_work.data()), N,
            reinterpret_cast<lapack_complex_double*>(S_work.data()), N,
            &W_all[static_cast<std::size_t>(N) * b]);
        if (info != 0) {
            std::fprintf(stderr, "LAPACKE_zhegvd failed for batch %d, info=%d\n", b, (int)info);
            return 3;
        }
        std::memcpy(&U_all[mat_elems * b], H_work.data(), sizeof(cdouble) * mat_elems);

        for (int i = 0; i < N; ++i) {
            double w = W_all[static_cast<std::size_t>(N) * b + i];
            evnorm_sq += w * w;
        }
    }

    // Self-consistency check: max relative residual across all eigenpairs.
    // For each (lambda_i, u_i) we compute
    //   r = ||H u_i - lambda_i S u_i||_inf / (||H||_F + |lambda_i| * ||S||_F)
    // which should be < ~1e-12 in fp64 for a well-conditioned problem.
    double max_rel_residual = -1.0;
    if (args.check_resid) {
        std::vector<cdouble> Hu(N), Su(N);
        for (int b = 0; b < B; ++b) {
            const cdouble* Hb = &H_all[mat_elems * b];
            const cdouble* Sb = &S_all[mat_elems * b];
            const cdouble* Ub = &U_all[mat_elems * b];
            const double*  Wb = &W_all[static_cast<std::size_t>(N) * b];

            // Frobenius norms of H and S.
            double Hf2 = 0.0, Sf2 = 0.0;
            for (std::size_t k = 0; k < mat_elems; ++k) {
                Hf2 += std::norm(Hb[k]);
                Sf2 += std::norm(Sb[k]);
            }
            const double Hf = std::sqrt(Hf2);
            const double Sf = std::sqrt(Sf2);

            for (int i = 0; i < N; ++i) {
                const cdouble* u  = &Ub[static_cast<std::size_t>(i) * N];
                const double  lam = Wb[i];

                // Hu = H * u, Su = S * u (column-major).
                for (int row = 0; row < N; ++row) {
                    cdouble hh(0.0, 0.0), ss(0.0, 0.0);
                    for (int col = 0; col < N; ++col) {
                        hh += Hb[static_cast<std::size_t>(col) * N + row] * u[col];
                        ss += Sb[static_cast<std::size_t>(col) * N + row] * u[col];
                    }
                    Hu[row] = hh;
                    Su[row] = ss;
                }

                double rmax = 0.0;
                for (int row = 0; row < N; ++row) {
                    double v = std::abs(Hu[row] - lam * Su[row]);
                    rmax = std::max(rmax, v);
                }
                const double denom = Hf + std::abs(lam) * Sf;
                const double rel = (denom > 0.0) ? (rmax / denom) : rmax;
                max_rel_residual = std::max(max_rel_residual, rel);
            }
        }
    }

    // Write the binary artifact.
    std::ofstream f(args.out, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "cannot open %s for writing\n", args.out.c_str());
        return 4;
    }
    auto write_i32 = [&](std::int32_t v) {
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    write_i32(kMagic);
    write_i32(kVersion);
    write_i32(N);
    write_i32(B);
    write_i32(args.seed);
    write_i32(0);  // reserved

    f.write(reinterpret_cast<const char*>(H_all.data()),
            static_cast<std::streamsize>(sizeof(cdouble) * H_all.size()));
    f.write(reinterpret_cast<const char*>(S_all.data()),
            static_cast<std::streamsize>(sizeof(cdouble) * S_all.size()));
    f.write(reinterpret_cast<const char*>(W_all.data()),
            static_cast<std::streamsize>(sizeof(double)  * W_all.size()));
    f.write(reinterpret_cast<const char*>(U_all.data()),
            static_cast<std::streamsize>(sizeof(cdouble) * U_all.size()));
    if (!f) {
        std::fprintf(stderr, "write to %s failed\n", args.out.c_str());
        return 5;
    }
    f.close();

    const char* mode_str = (args.s_mode == SMode::Random) ? "random" : "near-identity";
    std::printf(
        "LAPACK reference computed: N=%d, B=%d, seed=%d, s_mode=%s, output=%s, "
        "total eigenvalues_norm=%.6e",
        N, B, args.seed, mode_str, args.out.c_str(), std::sqrt(evnorm_sq));
    if (args.check_resid) {
        std::printf(", max_rel_residual=%.3e", max_rel_residual);
    }
    std::printf("\n");
    return 0;
}
