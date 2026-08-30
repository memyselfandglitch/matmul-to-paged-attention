#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cstring>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// The article uses eight 32-bit floats in one 256-bit AVX register.
// AArch64/NEON registers are 128-bit, so use four floats on this machine.
#if defined(__AVX__)
typedef float vec __attribute__ (( vector_size(32) ));
constexpr int SIMD_LANES = 8;
#else
typedef float vec __attribute__ (( vector_size(16) ));
constexpr int SIMD_LANES = 4;
#endif

int TILE_SIZE = 64;


using Clock = std::chrono::steady_clock;

// ----------------------------------------------------
// Baseline
// ----------------------------------------------------

void matmul(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i * n + j] +=
                    a[i * n + k] * b[k * n + j];
            }
        }
    }
}

// ----------------------------------------------------
// B already transposed
// ----------------------------------------------------

void matmul_transposed(
    const float *a,
    const float *bt,
    float *c,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i * n + j] +=
                    a[i * n + k] * bt[j * n + k];
            }
        }
    }
}

// ----------------------------------------------------
// Explicit SIMD, with B already transposed
// ----------------------------------------------------

void matmul_simd(
    const float *a,
    const float *bt,
    float *c,
    int n
) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vec sum{};
            int k = 0;

            for (; k + SIMD_LANES <= n; k += SIMD_LANES) {
                vec av;
                vec bv;

                // memcpy permits unaligned vector loads.
                std::memcpy(&av, &a[i * n + k], sizeof(vec));
                std::memcpy(&bv, &bt[j * n + k], sizeof(vec));
                sum += av * bv;
            }

            for (int lane = 0; lane < SIMD_LANES; lane++) {
                c[i * n + j] += sum[lane];
            }

            // Handle sizes that are not a multiple of SIMD_LANES.
            for (; k < n; k++) {
                c[i * n + j] +=
                    a[i * n + k] * bt[j * n + k];
            }
        }
    }
}

// ----------------------------------------------------
// Six-loop cache tiling + SIMD, with B transposed
// ----------------------------------------------------

void matmul_tiled_simd(
    const float *a,
    const float *bt,
    float *c,
    int n
) {
    for (int ii = 0; ii < n; ii += TILE_SIZE) {
        for (int jj = 0; jj < n; jj += TILE_SIZE) {
            for (int kk = 0; kk < n; kk += TILE_SIZE) {
                int i_end = std::min(ii + TILE_SIZE, n);
                int j_end = std::min(jj + TILE_SIZE, n);
                int k_end = std::min(kk + TILE_SIZE, n);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        vec sum{};
                        int k = kk;

                        for (; k + SIMD_LANES <= k_end;
                             k += SIMD_LANES) {
                            vec av;
                            vec bv;
                            std::memcpy(
                                &av, &a[i * n + k], sizeof(vec));
                            std::memcpy(
                                &bv, &bt[j * n + k], sizeof(vec));
                            sum += av * bv;
                        }

                        for (int lane = 0; lane < SIMD_LANES; lane++) {
                            c[i * n + j] += sum[lane];
                        }

                        for (; k < k_end; k++) {
                            c[i * n + j] +=
                                a[i * n + k] * bt[j * n + k];
                        }
                    }
                }
            }
        }
    }
}

std::size_t cache_size(const char *name) {
#if defined(__APPLE__)
    std::size_t value = 0;
    std::size_t value_size = sizeof(value);
    if (sysctlbyname(name, &value, &value_size, nullptr, 0) == 0) {
        return value;
    }
#else
    (void) name;
#endif
    return 0;
}

int tile_size_for_cache(std::size_t cache_bytes) {
    // Limit the three square tiles to 75% of L1, leaving room for other data.
    int tile = SIMD_LANES;
    while (true) {
        int next = tile + SIMD_LANES;
        std::size_t working_set =
            3ULL * next * next * sizeof(float);
        if (working_set > cache_bytes * 3 / 4) {
            return tile;
        }
        tile = next;
    }
}

// ----------------------------------------------------
// Benchmark one kernel
// ----------------------------------------------------

double run_once(
    void (*fn)(const float *, const float *, float *, int),
    const float *A,
    const float *B,
    float *C,
    int n
) {
    std::fill(C, C + n * n, 0.0f);

    auto start = Clock::now();

    fn(A, B, C, n);

    auto end = Clock::now();

    return std::chrono::duration<double, std::micro>(
        end - start
    ).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    size_t size = values.size();

    if (size % 2 == 1)
        return values[size / 2];

    return (values[size / 2 - 1] + values[size / 2]) / 2.0;
}

int main() {

    constexpr int WARMUPS = 2;
    constexpr int REPS = 10;

    // ------------------------------------------------
    // n values: all multiples of 48
    // ------------------------------------------------

    std::vector<int> sizes;

    for (int n = 48; n <= 1920; n += 48) {
        sizes.push_back(n);
    }

    // Randomize size order so increasing n doesn't
    // correlate with CPU temperature / frequency.
    std::random_device rd;
    std::mt19937 rng(rd());

    std::shuffle(sizes.begin(), sizes.end(), rng);

    std::uniform_real_distribution<float>
        dist(0.0f, 1.0f);

    std::size_t performance_l1 =
        cache_size("hw.perflevel0.l1dcachesize");
    std::size_t performance_l2 =
        cache_size("hw.perflevel0.l2cachesize");
    std::size_t efficiency_l1 =
        cache_size("hw.perflevel1.l1dcachesize");
    std::size_t efficiency_l2 =
        cache_size("hw.perflevel1.l2cachesize");

    std::size_t tiling_cache = 64 * 1024;
    if (performance_l1 != 0) {
        tiling_cache = performance_l1;
    }
    if (efficiency_l1 != 0) {
        tiling_cache = std::min(tiling_cache, efficiency_l1);
    }
    TILE_SIZE = tile_size_for_cache(tiling_cache);

    std::size_t tile_working_set =
        3ULL * TILE_SIZE * TILE_SIZE * sizeof(float);

    std::cout
        << "Performance-core cache: L1D " << performance_l1 / 1024
        << " KiB, L2 " << performance_l2 / (1024 * 1024) << " MiB\n"
        << "Efficiency-core cache: L1D " << efficiency_l1 / 1024
        << " KiB, L2 " << efficiency_l2 / (1024 * 1024) << " MiB\n"
        << "Selected tile: " << TILE_SIZE << "x" << TILE_SIZE
        << "; A + BT + C tile working set: "
        << tile_working_set / 1024 << " KiB\n\n";

    std::cout
        << "float is " << sizeof(float) * 8 << " bits; "
        << SIMD_LANES << " floats use " << sizeof(vec) * 8
        << " SIMD bits.\n";

#if defined(__aarch64__)
    std::cout
        << "Hardware: ARM NEON has 128-bit vector registers, not 256-bit "
        << "registers.\n";
#elif defined(__AVX__)
    std::cout
        << "Hardware target: 256-bit AVX is enabled for this build.\n";
#else
    std::cout
        << "Hardware target: 256-bit AVX is not enabled for this build.\n";
#endif

    std::cout
        << "8 float32 values = 256 bits; 8 float16 values = 128 bits.\n\n";

    std::cout
        << std::setw(8)  << "n"
        << std::setw(14) << "Memory(MB)"
        << std::setw(18) << "Baseline(us)"
        << std::setw(18) << "Transpose(us)"
        << std::setw(18) << "SIMD(us)"
        << std::setw(18) << "Tiled SIMD(us)"
        << std::setw(14) << "Base/BT"
        << std::setw(14) << "Base/SIMD"
        << std::setw(14) << "BT/SIMD"
        << std::setw(14) << "SIMD/Tiled"
        << "\n";

    // ------------------------------------------------
    // Test each matrix size
    // ------------------------------------------------

    for (int n : sizes) {

        std::vector<float> A(n * n);
        std::vector<float> B(n * n);
        std::vector<float> BT(n * n);
        std::vector<float> C(n * n);

        // Fresh random matrices for this n
        for (int i = 0; i < n * n; i++) {
            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        // Transpose B
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                BT[i * n + j] =
                    B[j * n + i];
            }
        }

        // --------------------------------------------
        // Warmups
        // --------------------------------------------

        for (int w = 0; w < WARMUPS; w++) {

            run_once(
                matmul,
                A.data(),
                B.data(),
                C.data(),
                n
            );

            run_once(
                matmul_transposed,
                A.data(),
                BT.data(),
                C.data(),
                n
            );

            run_once(
                matmul_simd,
                A.data(),
                BT.data(),
                C.data(),
                n
            );

            run_once(
                matmul_tiled_simd,
                A.data(),
                BT.data(),
                C.data(),
                n
            );
        }

        // --------------------------------------------
        // Measurements
        // --------------------------------------------

        std::vector<double> baseline_times;
        std::vector<double> transposed_times;
        std::vector<double> simd_times;
        std::vector<double> tiled_simd_times;

        for (int r = 0; r < REPS; r++) {

            // Alternate order to reduce "second-run"
            // cache / thermal bias.
            if (r % 2 == 0) {

                baseline_times.push_back(
                    run_once(
                        matmul,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );

                transposed_times.push_back(
                    run_once(
                        matmul_transposed,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                simd_times.push_back(
                    run_once(
                        matmul_simd,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                tiled_simd_times.push_back(
                    run_once(
                        matmul_tiled_simd,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

            } else {

                tiled_simd_times.push_back(
                    run_once(
                        matmul_tiled_simd,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                simd_times.push_back(
                    run_once(
                        matmul_simd,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                transposed_times.push_back(
                    run_once(
                        matmul_transposed,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                baseline_times.push_back(
                    run_once(
                        matmul,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );
            }
        }

        double baseline =
            median(baseline_times);

        double transposed =
            median(transposed_times);

        double simd =
            median(simd_times);

        double tiled_simd =
            median(tiled_simd_times);

        double transpose_speedup =
            baseline / transposed;

        double simd_speedup =
            baseline / simd;

        double simd_over_transposed =
            transposed / simd;

        double tiling_over_simd =
            simd / tiled_simd;

        // A + B + C
        double memory_mb =
            3.0 * n * n * sizeof(float)
            / (1024.0 * 1024.0);

        std::cout
            << std::setw(8)  << n
            << std::setw(14) << std::fixed << std::setprecision(2)
            << memory_mb
            << std::setw(18) << baseline
            << std::setw(18) << transposed
            << std::setw(18) << simd
            << std::setw(18) << tiled_simd
            << std::setw(13) << transpose_speedup << "x"
            << std::setw(13) << simd_speedup << "x"
            << std::setw(13) << simd_over_transposed << "x"
            << std::setw(13) << tiling_over_simd << "x"
            << "\n";
    }

    return 0;
}
