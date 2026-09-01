// Fair comparison of the six macro-tile loop orders for row-major SGEMM.
//
// Every order is measured at two optimization stages:
//   1. cache tiling with a compiler-vectorizable inner loop;
//   2. the same cache tiling with one shared SIMD register microkernel.
//
// Within a stage, only the order of the Mc, Nc, and Kc loops changes. Tile
// sizes, edge handling, arithmetic, initialization, and measurement are shared.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#define RESTRICT __restrict__
#else
#define NOINLINE
#define RESTRICT
#endif

// Use the native AVX width when the compiler target enables it. AArch64/NEON
// and non-AVX x86 builds use 128-bit vectors.
#if defined(__AVX__)
using Vec = float __attribute__((vector_size(32)));
#else
using Vec = float __attribute__((vector_size(16)));
#endif

constexpr int SIMD_LANES = sizeof(Vec) / sizeof(float);

// Macro tiles: M (rows), N (columns), and K (reduction depth).
constexpr int MC = 120;
constexpr int NC = 64;
constexpr int KC = 240;

// Register tile: six rows by two SIMD vectors.
constexpr int MR = 6;
constexpr int NR_BLOCKS = 2;
constexpr int NR = NR_BLOCKS * SIMD_LANES;

static_assert(MC % MR == 0, "MC must be a multiple of MR");
static_assert(NC % NR == 0, "NC must be a multiple of NR");

using Clock = std::chrono::steady_clock;
using Matrix = std::vector<float>;
using Kernel = void (*)(const float *, const float *, float *, int);

constexpr float MAX_RELATIVE_ERROR = 1.0e-3f;

// Prevent dead-code elimination of the timed matrix multiplication.
volatile float benchmark_sink = 0.0f;

// Row-major index of matrix[row][col].
inline std::size_t index_of(int row, int col, int n) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
           static_cast<std::size_t>(col);
}

inline Vec load_vec(const float *source) {
    Vec value;
    // memcpy permits an unaligned load without violating C++ aliasing rules.
    std::memcpy(&value, source, sizeof(value));
    return value;
}

inline void store_vec(float *destination, Vec value) {
    std::memcpy(destination, &value, sizeof(value));
}

inline Vec broadcast(float value) {
    Vec result{};
    for (int lane = 0; lane < SIMD_LANES; ++lane) {
        result[lane] = value;
    }
    return result;
}

enum class LoopOrder { IJK, IKJ, JIK, JKI, KIJ, KJI };

// Visit every (Mc, Nc, Kc) macro tile. `if constexpr` selects one loop nest at
// compile time, so there is no order switch or indirect call inside the loops.
template <LoopOrder Order, typename ProcessTile>
inline void for_each_macro_tile(int n, ProcessTile process_tile) {
    if constexpr (Order == LoopOrder::IJK) {
        for (int ic = 0; ic < n; ic += MC)
            for (int jc = 0; jc < n; jc += NC)
                for (int pc = 0; pc < n; pc += KC)
                    process_tile(ic, jc, pc);
    } else if constexpr (Order == LoopOrder::IKJ) {
        for (int ic = 0; ic < n; ic += MC)
            for (int pc = 0; pc < n; pc += KC)
                for (int jc = 0; jc < n; jc += NC)
                    process_tile(ic, jc, pc);
    } else if constexpr (Order == LoopOrder::JIK) {
        for (int jc = 0; jc < n; jc += NC)
            for (int ic = 0; ic < n; ic += MC)
                for (int pc = 0; pc < n; pc += KC)
                    process_tile(ic, jc, pc);
    } else if constexpr (Order == LoopOrder::JKI) {
        for (int jc = 0; jc < n; jc += NC)
            for (int pc = 0; pc < n; pc += KC)
                for (int ic = 0; ic < n; ic += MC)
                    process_tile(ic, jc, pc);
    } else if constexpr (Order == LoopOrder::KIJ) {
        for (int pc = 0; pc < n; pc += KC)
            for (int ic = 0; ic < n; ic += MC)
                for (int jc = 0; jc < n; jc += NC)
                    process_tile(ic, jc, pc);
    } else {
        static_assert(Order == LoopOrder::KJI);
        for (int pc = 0; pc < n; pc += KC)
            for (int jc = 0; jc < n; jc += NC)
                for (int ic = 0; ic < n; ic += MC)
                    process_tile(ic, jc, pc);
    }
}

// ---------------------------------------------------------------------------
// Stage 1: cache tiling
// ---------------------------------------------------------------------------

// The inner i-k-j traversal is identical for all six macro orders. Its j loop
// walks B and C contiguously and is suitable for compiler auto-vectorization.
inline void process_tiled_macro_tile(const float *RESTRICT a,
                                     const float *RESTRICT b,
                                     float *RESTRICT c, int n, int ic, int jc,
                                     int pc) {
    const int i_end = std::min(ic + MC, n);
    const int j_end = std::min(jc + NC, n);
    const int k_end = std::min(pc + KC, n);
    const int width = j_end - jc;

    for (int i = ic; i < i_end; ++i) {
        const float *a_row = &a[index_of(i, 0, n)];
        float *c_segment = &c[index_of(i, jc, n)];

        for (int k = pc; k < k_end; ++k) {
            const float a_ik = a_row[k];
            const float *b_segment = &b[index_of(k, jc, n)];

            for (int offset = 0; offset < width; ++offset) {
                c_segment[offset] += a_ik * b_segment[offset];
            }
        }
    }
}

template <LoopOrder Order>
NOINLINE void tiled_kernel(const float *RESTRICT a, const float *RESTRICT b,
                           float *RESTRICT c, int n) {
    for_each_macro_tile<Order>(n, [&](int ic, int jc, int pc) {
        process_tiled_macro_tile(a, b, c, n, ic, jc, pc);
    });
}

// ---------------------------------------------------------------------------
// Stage 2: cache tiling plus SIMD register blocking
// ---------------------------------------------------------------------------

// Compute an MR x NR output tile and keep all partial sums in registers over
// [k_begin, k_end). B vectors are shared by six output rows; each broadcast A
// value is shared by two output vectors.
inline void register_microkernel(const float *RESTRICT a,
                                 const float *RESTRICT b, float *RESTRICT c,
                                 int n, int row0, int col0, int k_begin,
                                 int k_end) {
    Vec accumulator[MR][NR_BLOCKS] = {};

    for (int k = k_begin; k < k_end; ++k) {
        Vec b_values[NR_BLOCKS];
        for (int column_block = 0; column_block < NR_BLOCKS; ++column_block) {
            const int col = col0 + column_block * SIMD_LANES;
            b_values[column_block] = load_vec(&b[index_of(k, col, n)]);
        }

        for (int row = 0; row < MR; ++row) {
            const Vec a_value = broadcast(a[index_of(row0 + row, k, n)]);
            for (int column_block = 0; column_block < NR_BLOCKS;
                 ++column_block) {
                accumulator[row][column_block] +=
                    a_value * b_values[column_block];
            }
        }
    }

    for (int row = 0; row < MR; ++row) {
        for (int column_block = 0; column_block < NR_BLOCKS; ++column_block) {
            const int col = col0 + column_block * SIMD_LANES;
            float *destination = &c[index_of(row0 + row, col, n)];
            store_vec(destination,
                      load_vec(destination) + accumulator[row][column_block]);
        }
    }
}

// Scalar cleanup for an incomplete MR x NR tile at a matrix edge.
inline void scalar_edge_microkernel(const float *RESTRICT a,
                                    const float *RESTRICT b,
                                    float *RESTRICT c, int n, int row_begin,
                                    int row_end, int col_begin, int col_end,
                                    int k_begin, int k_end) {
    for (int i = row_begin; i < row_end; ++i) {
        for (int j = col_begin; j < col_end; ++j) {
            float sum = c[index_of(i, j, n)];
            for (int k = k_begin; k < k_end; ++k) {
                sum += a[index_of(i, k, n)] * b[index_of(k, j, n)];
            }
            c[index_of(i, j, n)] = sum;
        }
    }
}

inline void process_register_macro_tile(const float *RESTRICT a,
                                        const float *RESTRICT b,
                                        float *RESTRICT c, int n, int ic,
                                        int jc, int pc) {
    const int i_end = std::min(ic + MC, n);
    const int j_end = std::min(jc + NC, n);
    const int k_end = std::min(pc + KC, n);

    for (int row = ic; row < i_end; row += MR) {
        for (int col = jc; col < j_end; col += NR) {
            const int row_end = std::min(row + MR, i_end);
            const int col_end = std::min(col + NR, j_end);

            if (row_end - row == MR && col_end - col == NR) {
                register_microkernel(a, b, c, n, row, col, pc, k_end);
            } else {
                scalar_edge_microkernel(a, b, c, n, row, row_end, col,
                                        col_end, pc, k_end);
            }
        }
    }
}

template <LoopOrder Order>
NOINLINE void register_blocked_kernel(const float *RESTRICT a,
                                      const float *RESTRICT b,
                                      float *RESTRICT c, int n) {
    for_each_macro_tile<Order>(n, [&](int ic, int jc, int pc) {
        process_register_macro_tile(a, b, c, n, ic, jc, pc);
    });
}

struct OrderKernels {
    const char *name;
    Kernel tiled;
    Kernel register_blocked;
};

constexpr std::array<OrderKernels, 6> ORDERS{{
    {"ijk", tiled_kernel<LoopOrder::IJK>,
     register_blocked_kernel<LoopOrder::IJK>},
    {"ikj", tiled_kernel<LoopOrder::IKJ>,
     register_blocked_kernel<LoopOrder::IKJ>},
    {"jik", tiled_kernel<LoopOrder::JIK>,
     register_blocked_kernel<LoopOrder::JIK>},
    {"jki", tiled_kernel<LoopOrder::JKI>,
     register_blocked_kernel<LoopOrder::JKI>},
    {"kij", tiled_kernel<LoopOrder::KIJ>,
     register_blocked_kernel<LoopOrder::KIJ>},
    {"kji", tiled_kernel<LoopOrder::KJI>,
     register_blocked_kernel<LoopOrder::KJI>},
}};

constexpr std::size_t NUM_ORDERS = ORDERS.size();

// ---------------------------------------------------------------------------
// Correctness and timing helpers
// ---------------------------------------------------------------------------

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

double gflops(int n, double microseconds) {
    const double dimension = static_cast<double>(n);
    return 2.0 * dimension * dimension * dimension /
           (microseconds * 1.0e3);
}

float relative_error(const Matrix &expected, const Matrix &actual) {
    float worst = 0.0f;
    for (std::size_t position = 0; position < expected.size(); ++position) {
        const float denominator = std::max(1.0f, std::abs(expected[position]));
        worst = std::max(
            worst,
            std::abs(expected[position] - actual[position]) / denominator);
    }
    return worst;
}

void scalar_reference(const Matrix &a, const Matrix &b, Matrix &c, int n) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                c[index_of(i, j, n)] +=
                    a[index_of(i, k, n)] * b[index_of(k, j, n)];
}

// C initialization and the anti-optimization sink are outside the timed span.
double run_once(Kernel kernel, const Matrix &a, const Matrix &b, Matrix &c,
                int n) {
    std::fill(c.begin(), c.end(), 0.0f);
    const auto start = Clock::now();
    kernel(a.data(), b.data(), c.data(), n);
    const auto end = Clock::now();
    benchmark_sink = benchmark_sink + c[c.size() / 2];
    return std::chrono::duration<double, std::micro>(end - start).count();
}

void check_result(const char *order, const char *stage, const Matrix &expected,
                  const Matrix &actual, int n) {
    const float error = relative_error(expected, actual);
    if (error > MAX_RELATIVE_ERROR) {
        throw std::runtime_error(std::string(order) + " " + stage +
                                 " failed validation at n=" +
                                 std::to_string(n) + ", relative error=" +
                                 std::to_string(error));
    }
}

// n=70 is intentionally not aligned to the macro or register tile dimensions.
void validate_edge_paths(std::mt19937 &rng) {
    constexpr int n = 70;
    const std::size_t elements = static_cast<std::size_t>(n) * n;
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    Matrix a(elements), b(elements), expected(elements), actual(elements);
    for (std::size_t position = 0; position < elements; ++position) {
        a[position] = distribution(rng);
        b[position] = distribution(rng);
    }

    scalar_reference(a, b, expected, n);
    for (const OrderKernels &order : ORDERS) {
        run_once(order.tiled, a, b, actual, n);
        check_result(order.name, "tiled", expected, actual, n);
        run_once(order.register_blocked, a, b, actual, n);
        check_result(order.name, "register-blocked", expected, actual, n);
    }
}

// Run every measured kernel once before timing and cross-check their results.
void warm_up_and_cross_check(const Matrix &a, const Matrix &b, Matrix &c,
                             int n) {
    Matrix expected;
    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        const OrderKernels &order = ORDERS[order_index];

        run_once(order.tiled, a, b, c, n);
        if (order_index == 0) {
            expected = c;
        } else {
            check_result(order.name, "tiled", expected, c, n);
        }

        run_once(order.register_blocked, a, b, c, n);
        check_result(order.name, "register-blocked", expected, c, n);
    }
}

struct Timings {
    std::array<double, NUM_ORDERS> tiled{};
    std::array<double, NUM_ORDERS> register_blocked{};
};

struct BenchmarkTask {
    std::size_t order_index;
    bool register_blocked;
};

// Shuffle all twelve order/stage combinations on every repetition to reduce
// systematic cache, CPU-frequency, and thermal bias.
Timings time_all_kernels(const Matrix &a, const Matrix &b, Matrix &c, int n,
                         int repetitions, std::mt19937 &rng) {
    std::array<std::vector<double>, NUM_ORDERS> tiled_samples;
    std::array<std::vector<double>, NUM_ORDERS> register_samples;
    std::array<BenchmarkTask, 2 * NUM_ORDERS> tasks{};

    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        tasks[2 * order_index] = {order_index, false};
        tasks[2 * order_index + 1] = {order_index, true};
    }

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        std::shuffle(tasks.begin(), tasks.end(), rng);
        for (const BenchmarkTask &task : tasks) {
            const OrderKernels &order = ORDERS[task.order_index];
            if (task.register_blocked) {
                register_samples[task.order_index].push_back(
                    run_once(order.register_blocked, a, b, c, n));
            } else {
                tiled_samples[task.order_index].push_back(
                    run_once(order.tiled, a, b, c, n));
            }
        }
    }

    Timings result;
    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        result.tiled[order_index] =
            median_of(tiled_samples[order_index]);
        result.register_blocked[order_index] =
            median_of(register_samples[order_index]);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void print_table_header() {
    std::cout << std::setw(7) << "n" << std::setw(8) << "order"
              << std::setw(14) << "tiled(us)" << std::setw(13) << "tiled GF/s"
              << std::setw(14) << "t+reg(us)" << std::setw(13) << "t+reg GF/s"
              << std::setw(15) << "tile/t+reg" << std::setw(15)
              << "t+reg vs ijk" << std::setw(15) << "t+reg/best" << '\n';
}

void print_size_rows(int n, const Timings &timings) {
    const double best_register = *std::min_element(
        timings.register_blocked.begin(), timings.register_blocked.end());

    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        const double tiled = timings.tiled[order_index];
        const double reg = timings.register_blocked[order_index];
        std::cout << std::setw(7) << n << std::setw(8)
                  << ORDERS[order_index].name << std::fixed
                  << std::setprecision(2) << std::setw(14) << tiled
                  << std::setw(13) << gflops(n, tiled) << std::setw(14) << reg
                  << std::setw(13) << gflops(n, reg) << std::setw(14)
                  << tiled / reg << "x" << std::setw(14)
                  << timings.register_blocked[0] / reg << "x"
                  << std::setw(14) << reg / best_register << "x\n";
    }
    std::cout << '\n';
}

struct Summary {
    std::array<double, NUM_ORDERS> tiled_log_slowdown{};
    std::array<double, NUM_ORDERS> register_log_slowdown{};
    std::array<double, NUM_ORDERS> speedup_log_sum{};
    std::array<int, NUM_ORDERS> tiled_wins{};
    std::array<int, NUM_ORDERS> register_wins{};
};

void add_to_summary(Summary &summary, const Timings &timings) {
    const double best_tiled =
        *std::min_element(timings.tiled.begin(), timings.tiled.end());
    const double best_register = *std::min_element(
        timings.register_blocked.begin(), timings.register_blocked.end());

    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        const double tiled = timings.tiled[order_index];
        const double reg = timings.register_blocked[order_index];
        summary.tiled_log_slowdown[order_index] += std::log(tiled / best_tiled);
        summary.register_log_slowdown[order_index] +=
            std::log(reg / best_register);
        summary.speedup_log_sum[order_index] += std::log(tiled / reg);
        if (tiled == best_tiled) ++summary.tiled_wins[order_index];
        if (reg == best_register) ++summary.register_wins[order_index];
    }
}

void print_summary(const Summary &summary, std::size_t number_of_sizes) {
    std::cout << "Geometric means across tested matrix sizes:\n"
              << std::setw(8) << "order" << std::setw(16) << "tiled/best"
              << std::setw(13) << "tile wins" << std::setw(16)
              << "t+reg/best" << std::setw(12) << "t+r wins"
              << std::setw(18) << "tile/t+reg" << '\n';

    for (std::size_t order_index = 0; order_index < NUM_ORDERS;
         ++order_index) {
        const double divisor = static_cast<double>(number_of_sizes);
        std::cout << std::setw(8) << ORDERS[order_index].name << std::fixed
                  << std::setprecision(3) << std::setw(15)
                  << std::exp(summary.tiled_log_slowdown[order_index] / divisor)
                  << "x" << std::setw(13) << summary.tiled_wins[order_index]
                  << std::setw(15)
                  << std::exp(summary.register_log_slowdown[order_index] /
                              divisor)
                  << "x" << std::setw(12)
                  << summary.register_wins[order_index] << std::setw(17)
                  << std::exp(summary.speedup_log_sum[order_index] / divisor)
                  << "x\n";
    }
}

// ---------------------------------------------------------------------------
// Command-line handling and main
// ---------------------------------------------------------------------------

struct Options {
    int max_n = 1920;
    int repetitions = 5;
    int min_n = 384;
};

int parse_positive_int(const char *text, const char *name) {
    try {
        const int value = std::stoi(text);
        if (value <= 0) throw std::invalid_argument("not positive");
        return value;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive integer");
    }
}

Options parse_options(int argc, char **argv) {
    if (argc > 4) {
        throw std::invalid_argument(std::string("Usage: ") + argv[0] +
                                    " [max_n=1920] [repetitions=5] "
                                    "[min_n=384]");
    }

    Options options;
    if (argc > 1) options.max_n = parse_positive_int(argv[1], "max_n");
    if (argc > 2)
        options.repetitions = parse_positive_int(argv[2], "repetitions");
    if (argc > 3) options.min_n = parse_positive_int(argv[3], "min_n");
    if (options.min_n > options.max_n) {
        throw std::invalid_argument("min_n must not exceed max_n");
    }
    return options;
}

std::vector<int> choose_sizes(int min_n, int max_n) {
    constexpr std::array<int, 5> candidates{384, 768, 1152, 1536, 1920};
    std::vector<int> sizes;
    for (int n : candidates) {
        if (n >= min_n && n <= max_n) sizes.push_back(n);
    }
    if (sizes.empty() || sizes.back() != max_n) sizes.push_back(max_n);
    return sizes;
}

void print_banner() {
    std::cout << "Loop-order study for C += A * B (row-major float32)\n"
              << "SIMD: " << sizeof(Vec) * 8 << " bits, " << SIMD_LANES
              << " float lanes\n"
              << "Macro tiles (Mc x Nc x Kc): " << MC << " x " << NC
              << " x " << KC << '\n'
              << "Register tile (Mr x Nr): " << MR << " x " << NR << '\n'
              << "The named order applies only to Mc/Nc/Kc; both stages use "
                 "identical inner work.\n\n";
}

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        print_banner();

        std::mt19937 rng(0x5EEDu);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        validate_edge_paths(rng);

        const std::vector<int> sizes =
            choose_sizes(options.min_n, options.max_n);
        Summary summary;

        print_table_header();
        for (int n : sizes) {
            const std::size_t elements = static_cast<std::size_t>(n) * n;
            Matrix a(elements), b(elements), c(elements);
            for (std::size_t position = 0; position < elements; ++position) {
                a[position] = distribution(rng);
                b[position] = distribution(rng);
            }

            warm_up_and_cross_check(a, b, c, n);
            const Timings timings = time_all_kernels(
                a, b, c, n, options.repetitions, rng);
            add_to_summary(summary, timings);
            print_size_rows(n, timings);
        }

        print_summary(summary, sizes.size());
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
