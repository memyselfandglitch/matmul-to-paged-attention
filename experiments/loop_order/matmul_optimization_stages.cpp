// Benchmarks a progression of matrix-multiplication optimizations.
// The loop-order comparison lives in loop_order_study.cpp.

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <string>

#if defined(__AVX__)
typedef float vec __attribute__ (( vector_size(32) ));
#else
typedef float vec __attribute__ (( vector_size(16) ));
#endif

constexpr int SIMD_LANES = sizeof(vec) / sizeof(float);
constexpr int MICRO_ROWS = 6;
constexpr int MICRO_COLS = 2 * SIMD_LANES;
constexpr int TILE_SIZE = 64;
constexpr int COLUMN_TILE = 64;
constexpr int ROW_TILE = 120;
constexpr int DEPTH_TILE = 240;


using Clock = std::chrono::steady_clock;
volatile float benchmark_sink = 0.0f;

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

inline vec load_vec(const float *p) {
    vec value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

inline void store_vec(float *p, vec value) {
    std::memcpy(p, &value, sizeof(value));
}

inline vec broadcast(float value) {
    vec result{};
    for (int lane = 0; lane < SIMD_LANES; lane++) {
        result[lane] = value;
    }
    return result;
}

inline float horizontal_sum(vec value) {
    float result = 0.0f;
    for (int lane = 0; lane < SIMD_LANES; lane++) {
        result += value[lane];
    }
    return result;
}

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
                sum += load_vec(&a[i * n + k]) *
                       load_vec(&bt[j * n + k]);
            }

            float scalar_sum = horizontal_sum(sum);
            for (; k < n; k++) {
                scalar_sum += a[i * n + k] * bt[j * n + k];
            }
            c[i * n + j] += scalar_sum;
        }
    }
}

// ----------------------------------------------------
// Cache tiling + dot-product SIMD
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
                const int i_end = std::min(ii + TILE_SIZE, n);
                const int j_end = std::min(jj + TILE_SIZE, n);
                const int k_end = std::min(kk + TILE_SIZE, n);

                for (int i = ii; i < i_end; i++) {
                    for (int j = jj; j < j_end; j++) {
                        vec sum{};
                        int k = kk;

                        for (; k + SIMD_LANES <= k_end;
                             k += SIMD_LANES) {
                            sum += load_vec(&a[i * n + k]) *
                                   load_vec(&bt[j * n + k]);
                        }

                        float scalar_sum = horizontal_sum(sum);
                        for (; k < k_end; k++) {
                            scalar_sum +=
                                a[i * n + k] * bt[j * n + k];
                        }
                        c[i * n + j] += scalar_sum;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------
// Article-style register microkernel
// ----------------------------------------------------

inline void register_microkernel(
    const float *a,
    const float *b,
    float *c,
    int n,
    int x,
    int y,
    int k_begin,
    int k_end
) {
    // Six rows by two vectors: 6x16 on AVX, 6x8 on NEON.
    vec t00{}, t01{};
    vec t10{}, t11{};
    vec t20{}, t21{};
    vec t30{}, t31{};
    vec t40{}, t41{};
    vec t50{}, t51{};

    for (int k = k_begin; k < k_end; k++) {
        const vec b0 = load_vec(&b[k * n + y]);
        const vec b1 = load_vec(&b[k * n + y + SIMD_LANES]);

        const vec a0 = broadcast(a[(x + 0) * n + k]);
        t00 += a0 * b0;
        t01 += a0 * b1;

        const vec a1 = broadcast(a[(x + 1) * n + k]);
        t10 += a1 * b0;
        t11 += a1 * b1;

        const vec a2 = broadcast(a[(x + 2) * n + k]);
        t20 += a2 * b0;
        t21 += a2 * b1;

        const vec a3 = broadcast(a[(x + 3) * n + k]);
        t30 += a3 * b0;
        t31 += a3 * b1;

        const vec a4 = broadcast(a[(x + 4) * n + k]);
        t40 += a4 * b0;
        t41 += a4 * b1;

        const vec a5 = broadcast(a[(x + 5) * n + k]);
        t50 += a5 * b0;
        t51 += a5 * b1;
    }

    store_vec(&c[(x + 0) * n + y],
              load_vec(&c[(x + 0) * n + y]) + t00);
    store_vec(&c[(x + 0) * n + y + SIMD_LANES],
              load_vec(&c[(x + 0) * n + y + SIMD_LANES]) + t01);
    store_vec(&c[(x + 1) * n + y],
              load_vec(&c[(x + 1) * n + y]) + t10);
    store_vec(&c[(x + 1) * n + y + SIMD_LANES],
              load_vec(&c[(x + 1) * n + y + SIMD_LANES]) + t11);
    store_vec(&c[(x + 2) * n + y],
              load_vec(&c[(x + 2) * n + y]) + t20);
    store_vec(&c[(x + 2) * n + y + SIMD_LANES],
              load_vec(&c[(x + 2) * n + y + SIMD_LANES]) + t21);
    store_vec(&c[(x + 3) * n + y],
              load_vec(&c[(x + 3) * n + y]) + t30);
    store_vec(&c[(x + 3) * n + y + SIMD_LANES],
              load_vec(&c[(x + 3) * n + y + SIMD_LANES]) + t31);
    store_vec(&c[(x + 4) * n + y],
              load_vec(&c[(x + 4) * n + y]) + t40);
    store_vec(&c[(x + 4) * n + y + SIMD_LANES],
              load_vec(&c[(x + 4) * n + y + SIMD_LANES]) + t41);
    store_vec(&c[(x + 5) * n + y],
              load_vec(&c[(x + 5) * n + y]) + t50);
    store_vec(&c[(x + 5) * n + y + SIMD_LANES],
              load_vec(&c[(x + 5) * n + y + SIMD_LANES]) + t51);
}

void scalar_microkernel(
    const float *a,
    const float *b,
    float *c,
    int n,
    int x_begin,
    int x_end,
    int y_begin,
    int y_end,
    int k_begin,
    int k_end
) {
    for (int i = x_begin; i < x_end; i++) {
        for (int j = y_begin; j < y_end; j++) {
            float sum = c[i * n + j];
            for (int k = k_begin; k < k_end; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void run_microkernel_block(
    const float *a,
    const float *b,
    float *c,
    int n,
    int x,
    int x_end,
    int y,
    int y_end,
    int k_begin,
    int k_end
) {
    if (x + MICRO_ROWS <= x_end && y + MICRO_COLS <= y_end) {
        register_microkernel(a, b, c, n, x, y, k_begin, k_end);
    } else {
        scalar_microkernel(a, b, c, n,
                           x, std::min(x + MICRO_ROWS, x_end),
                           y, std::min(y + MICRO_COLS, y_end),
                           k_begin, k_end);
    }
}

void matmul_register_blocked_simd(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    for (int x = 0; x < n; x += MICRO_ROWS) {
        for (int y = 0; y < n; y += MICRO_COLS) {
            run_microkernel_block(a, b, c, n,
                                  x, n, y, n, 0, n);
        }
    }
}

void matmul_tiled_register_blocked_simd(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    // Macro-kernel order from the article: columns, rows, reduction.
    for (int jj = 0; jj < n; jj += COLUMN_TILE) {
        for (int ii = 0; ii < n; ii += ROW_TILE) {
            for (int kk = 0; kk < n; kk += DEPTH_TILE) {
                const int i_end = std::min(ii + ROW_TILE, n);
                const int j_end = std::min(jj + COLUMN_TILE, n);
                const int k_end = std::min(kk + DEPTH_TILE, n);

                for (int x = ii; x < i_end; x += MICRO_ROWS) {
                    for (int y = jj; y < j_end; y += MICRO_COLS) {
                        run_microkernel_block(a, b, c, n,
                                              x, i_end, y, j_end,
                                              kk, k_end);
                    }
                }
            }
        }
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
    benchmark_sink = benchmark_sink + C[(n * n) / 2];

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

double gflops(int n, double microseconds) {
    return 2.0 * n * n * n / (microseconds * 1.0e3);
}

float relative_error(
    const std::vector<float> &expected,
    const float *actual
) {
    float maximum = 0.0f;
    for (std::size_t index = 0; index < expected.size(); index++) {
        maximum = std::max(
            maximum,
            std::abs(expected[index] - actual[index]) /
                std::max(1.0f, std::abs(expected[index]))
        );
    }
    return maximum;
}

int main(int argc, char **argv) {

    constexpr int WARMUPS = 2;
    const int max_n = argc > 1 ? std::max(48, std::stoi(argv[1])) : 1920;
    const int reps = argc > 2 ? std::max(1, std::stoi(argv[2])) : 10;
    const int min_n = argc > 3 ? std::max(48, std::stoi(argv[3])) : 48;
    if (min_n > max_n) {
        std::cerr << "min_n must not exceed max_n\n";
        return 2;
    }

    // ------------------------------------------------
    // n values: all multiples of 48
    // ------------------------------------------------

    std::vector<int> sizes;

    for (int n = min_n; n <= max_n; n += 48) {
        sizes.push_back(n);
    }
    if (sizes.back() != max_n) {
        sizes.push_back(max_n);
    }

    // Randomize size order so increasing n doesn't
    // correlate with CPU temperature / frequency.
    std::random_device rd;
    std::mt19937 rng(rd());

    std::shuffle(sizes.begin(), sizes.end(), rng);

    std::uniform_real_distribution<float>
        dist(0.0f, 1.0f);

    std::cout
        << "SIMD: " << sizeof(vec) * 8 << " bits, "
        << SIMD_LANES << " float lanes\n"
        << "Register microkernel: " << MICRO_ROWS << "x" << MICRO_COLS
        << "; square tile: " << TILE_SIZE << "x" << TILE_SIZE << "\n"
        << "Article macro tiles (columns x rows x depth): "
        << COLUMN_TILE << "x" << ROW_TILE << "x" << DEPTH_TILE << "\n"
        << "B transpose preparation is excluded from timings.\n\n"
        << std::setw(8)  << "n"
        << std::setw(13) << "Memory(MB)"
        << std::setw(15) << "Baseline(us)"
        << std::setw(15) << "BT(us)"
        << std::setw(15) << "SIMD(us)"
        << std::setw(15) << "Tiled(us)"
        << std::setw(15) << "RegBlock(us)"
        << std::setw(15) << "Tile+Reg(us)"
        << std::setw(14) << "SIMD/Tiled"
        << std::setw(14) << "SIMD/Reg"
        << std::setw(14) << "Reg/Comb"
        << std::setw(14) << "SIMD/Comb"
        << std::setw(14) << "Comb GF/s"
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

        std::vector<float> reference;
        for (int w = 0; w < WARMUPS; w++) {

            run_once(
                matmul,
                A.data(),
                B.data(),
                C.data(),
                n
            );
            if (w == 0) {
                reference = C;
            }

            run_once(
                matmul_transposed,
                A.data(),
                BT.data(),
                C.data(),
                n
            );
            if (w == 0 && relative_error(reference, C.data()) > 1.0e-3f) {
                std::cerr << "transposed failed validation at n=" << n << '\n';
                return 2;
            }

            run_once(
                matmul_simd,
                A.data(),
                BT.data(),
                C.data(),
                n
            );
            if (w == 0 && relative_error(reference, C.data()) > 1.0e-3f) {
                std::cerr << "SIMD failed validation at n=" << n << '\n';
                return 2;
            }

            run_once(
                matmul_tiled_simd,
                A.data(),
                BT.data(),
                C.data(),
                n
            );
            if (w == 0 && relative_error(reference, C.data()) > 1.0e-3f) {
                std::cerr << "tiled SIMD failed validation at n=" << n << '\n';
                return 2;
            }

            run_once(
                matmul_register_blocked_simd,
                A.data(),
                B.data(),
                C.data(),
                n
            );
            if (w == 0 && relative_error(reference, C.data()) > 1.0e-3f) {
                std::cerr << "register block failed validation at n=" << n << '\n';
                return 2;
            }

            run_once(
                matmul_tiled_register_blocked_simd,
                A.data(),
                B.data(),
                C.data(),
                n
            );
            if (w == 0 && relative_error(reference, C.data()) > 1.0e-3f) {
                std::cerr << "tiled register block failed validation at n="
                          << n << '\n';
                return 2;
            }
        }

        // --------------------------------------------
        // Measurements
        // --------------------------------------------

        std::vector<double> baseline_times;
        std::vector<double> transposed_times;
        std::vector<double> simd_times;
        std::vector<double> tiled_times;
        std::vector<double> register_blocked_times;
        std::vector<double> tiled_register_blocked_times;

        for (int r = 0; r < reps; r++) {

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

                tiled_times.push_back(
                    run_once(
                        matmul_tiled_simd,
                        A.data(),
                        BT.data(),
                        C.data(),
                        n
                    )
                );

                register_blocked_times.push_back(
                    run_once(
                        matmul_register_blocked_simd,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );

                tiled_register_blocked_times.push_back(
                    run_once(
                        matmul_tiled_register_blocked_simd,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );

            } else {

                tiled_register_blocked_times.push_back(
                    run_once(
                        matmul_tiled_register_blocked_simd,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );

                register_blocked_times.push_back(
                    run_once(
                        matmul_register_blocked_simd,
                        A.data(),
                        B.data(),
                        C.data(),
                        n
                    )
                );

                tiled_times.push_back(
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

        double tiled =
            median(tiled_times);

        double register_blocked =
            median(register_blocked_times);

        double tiled_register_blocked =
            median(tiled_register_blocked_times);

        // A + B + C
        double memory_mb =
            3.0 * n * n * sizeof(float)
            / (1024.0 * 1024.0);

        std::cout
            << std::setw(8)  << n
            << std::setw(13) << std::fixed << std::setprecision(2)
            << memory_mb
            << std::setw(15) << baseline
            << std::setw(15) << transposed
            << std::setw(15) << simd
            << std::setw(15) << tiled
            << std::setw(15) << register_blocked
            << std::setw(15) << tiled_register_blocked
            << std::setw(13) << simd / tiled << "x"
            << std::setw(13) << simd / register_blocked << "x"
            << std::setw(13) << register_blocked / tiled_register_blocked << "x"
            << std::setw(13) << simd / tiled_register_blocked << "x"
            << std::setw(14) << gflops(n, tiled_register_blocked)
            << "\n";
    }

    return 0;
}
