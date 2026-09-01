// A small structural study of matrix multiplication, batched matrix
// multiplication, and the two batched products used with a K/V cache.
//
// This is deliberately portable C++, not a replacement for BLAS. The point is
// to make the batch dimension and its dispatch cost visible.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
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

using Clock = std::chrono::steady_clock;
using Matrix = std::vector<float>;

volatile float benchmark_sink = 0.0f;

inline std::size_t matrix_size(int rows, int columns) {
    return static_cast<std::size_t>(rows) * columns;
}

// C[M,N] += A[M,K] * B[K,N]. The i-k-j order walks B and C contiguously.
inline void gemm_impl(const float *RESTRICT a, const float *RESTRICT b,
                      float *RESTRICT c, int m, int n, int k_size) {
    for (int i = 0; i < m; ++i) {
        float *c_row = c + static_cast<std::size_t>(i) * n;
        const float *a_row = a + static_cast<std::size_t>(i) * k_size;
        for (int k = 0; k < k_size; ++k) {
            const float a_ik = a_row[k];
            const float *b_row = b + static_cast<std::size_t>(k) * n;
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// Represents a separately dispatched GEMM call.
NOINLINE void gemm_call(const float *a, const float *b, float *c, int m, int n,
                        int k_size) {
    gemm_impl(a, b, c, m, n, k_size);
}

// B separate GEMM calls. A library/user loop pays one dispatch per matrix.
NOINLINE void loop_of_gemms(const float *a, const float *b, float *c, int batch,
                            int m, int n, int k_size) {
    const std::size_t a_stride = matrix_size(m, k_size);
    const std::size_t b_stride = matrix_size(k_size, n);
    const std::size_t c_stride = matrix_size(m, n);

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        gemm_call(a + batch_index * a_stride, b + batch_index * b_stride,
                  c + batch_index * c_stride, m, n, k_size);
    }
}

// One BMM call. Mathematically this performs exactly the same independent
// GEMMs; internally, the implementation owns the batch loop and can schedule
// all matrices together.
NOINLINE void batch_matmul(const float *a, const float *b, float *c, int batch,
                           int m, int n, int k_size) {
    const std::size_t a_stride = matrix_size(m, k_size);
    const std::size_t b_stride = matrix_size(k_size, n);
    const std::size_t c_stride = matrix_size(m, n);

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        gemm_impl(a + batch_index * a_stride, b + batch_index * b_stride,
                  c + batch_index * c_stride, m, n, k_size);
    }
}

// If every batch item shares the same B, [batch,M,K] is contiguous and can be
// viewed as [batch*M,K]. This special case becomes one ordinary, larger GEMM.
NOINLINE void loop_with_shared_rhs(const float *a, const float *shared_b,
                                   float *c, int batch, int m, int n,
                                   int k_size) {
    const std::size_t a_stride = matrix_size(m, k_size);
    const std::size_t c_stride = matrix_size(m, n);
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        gemm_call(a + batch_index * a_stride, shared_b,
                  c + batch_index * c_stride, m, n, k_size);
    }
}

NOINLINE void flattened_shared_rhs(const float *a, const float *shared_b,
                                   float *c, int batch, int m, int n,
                                   int k_size) {
    gemm_call(a, shared_b, c, batch * m, n, k_size);
}

// ---------------------------------------------------------------------------
// K/V-cache attention
// ---------------------------------------------------------------------------

// One group is one (batch item, attention head). Shapes are:
//   Q      [query_tokens, head_dim]
//   K/V    [context_tokens, head_dim]
//   scores [query_tokens, context_tokens]
//   output [query_tokens, head_dim]
//
// It performs Q*K^T, softmax, and probabilities*V.
inline void attention_group_impl(const float *RESTRICT query,
                                 const float *RESTRICT key_cache,
                                 const float *RESTRICT value_cache,
                                 float *RESTRICT scores,
                                 float *RESTRICT output, int query_tokens,
                                 int context_tokens, int head_dim) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int q = 0; q < query_tokens; ++q) {
        const float *q_row = query + static_cast<std::size_t>(q) * head_dim;
        float *score_row =
            scores + static_cast<std::size_t>(q) * context_tokens;

        // First batched product: Q[1,D] * K^T[D,T] -> scores[1,T].
        for (int token = 0; token < context_tokens; ++token) {
            const float *k_row =
                key_cache + static_cast<std::size_t>(token) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
            score_row[token] = dot * scale;
        }

        // Stable softmax over cached tokens.
        const float maximum =
            *std::max_element(score_row, score_row + context_tokens);
        float denominator = 0.0f;
        for (int token = 0; token < context_tokens; ++token) {
            score_row[token] = std::exp(score_row[token] - maximum);
            denominator += score_row[token];
        }
        for (int token = 0; token < context_tokens; ++token) {
            score_row[token] /= denominator;
        }

        // Second batched product: probabilities[1,T] * V[T,D] -> output[1,D].
        float *output_row = output + static_cast<std::size_t>(q) * head_dim;
        std::fill(output_row, output_row + head_dim, 0.0f);
        for (int token = 0; token < context_tokens; ++token) {
            const float probability = score_row[token];
            const float *v_row =
                value_cache + static_cast<std::size_t>(token) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                output_row[d] += probability * v_row[d];
            }
        }
    }
}

NOINLINE void attention_group_call(const float *query, const float *key_cache,
                                   const float *value_cache, float *scores,
                                   float *output, int query_tokens,
                                   int context_tokens, int head_dim) {
    attention_group_impl(query, key_cache, value_cache, scores, output,
                         query_tokens, context_tokens, head_dim);
}

// Separate attention calls for every (batch, head) group.
NOINLINE void attention_looped(const float *query, const float *key_cache,
                               const float *value_cache, float *scores,
                               float *output, int groups, int query_tokens,
                               int context_tokens, int head_dim) {
    const std::size_t query_stride = matrix_size(query_tokens, head_dim);
    const std::size_t cache_stride = matrix_size(context_tokens, head_dim);
    const std::size_t score_stride =
        matrix_size(query_tokens, context_tokens);
    const std::size_t output_stride = matrix_size(query_tokens, head_dim);

    for (int group = 0; group < groups; ++group) {
        attention_group_call(
            query + group * query_stride, key_cache + group * cache_stride,
            value_cache + group * cache_stride, scores + group * score_stride,
            output + group * output_stride, query_tokens, context_tokens,
            head_dim);
    }
}

// One batched attention call. Batch and head are flattened into `groups`.
NOINLINE void attention_batched(const float *query, const float *key_cache,
                                const float *value_cache, float *scores,
                                float *output, int groups, int query_tokens,
                                int context_tokens, int head_dim) {
    const std::size_t query_stride = matrix_size(query_tokens, head_dim);
    const std::size_t cache_stride = matrix_size(context_tokens, head_dim);
    const std::size_t score_stride =
        matrix_size(query_tokens, context_tokens);
    const std::size_t output_stride = matrix_size(query_tokens, head_dim);

    for (int group = 0; group < groups; ++group) {
        attention_group_impl(
            query + group * query_stride, key_cache + group * cache_stride,
            value_cache + group * cache_stride, scores + group * score_stride,
            output + group * output_stride, query_tokens, context_tokens,
            head_dim);
    }
}

// ---------------------------------------------------------------------------
// Benchmark helpers and main
// ---------------------------------------------------------------------------

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 1) return samples[middle];
    return (samples[middle - 1] + samples[middle]) / 2.0;
}

template <typename Reset, typename Work>
double measure(Reset reset, Work work, const Matrix &observed_output,
               int repetitions) {
    std::vector<double> samples;
    samples.reserve(repetitions);

    reset();
    work();  // warm-up
    benchmark_sink = benchmark_sink +
                     observed_output[observed_output.size() / 2];

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        reset();
        const auto start = Clock::now();
        work();
        const auto end = Clock::now();
        benchmark_sink = benchmark_sink +
                         observed_output[observed_output.size() / 2];
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }
    return median(samples);
}

float maximum_relative_error(const Matrix &expected, const Matrix &actual) {
    float maximum = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float scale = std::max(1.0f, std::abs(expected[i]));
        maximum =
            std::max(maximum, std::abs(expected[i] - actual[i]) / scale);
    }
    return maximum;
}

void require_equal(const char *name, const Matrix &expected,
                   const Matrix &actual) {
    const float error = maximum_relative_error(expected, actual);
    if (error > 1.0e-4f) {
        throw std::runtime_error(std::string(name) +
                                 " failed validation; relative error=" +
                                 std::to_string(error));
    }
}

double gemm_gflops(int batch, int m, int n, int k_size,
                   double microseconds) {
    return 2.0 * batch * m * n * k_size / (microseconds * 1.0e3);
}

int parse_repetitions(int argc, char **argv) {
    if (argc > 2) {
        throw std::invalid_argument(std::string("Usage: ") + argv[0] +
                                    " [repetitions=9]");
    }
    if (argc == 1) return 9;
    const int value = std::stoi(argv[1]);
    if (value <= 0) throw std::invalid_argument("repetitions must be positive");
    return value;
}

void fill_random(Matrix &values, std::mt19937 &rng) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : values) value = distribution(rng);
}

int main(int argc, char **argv) {
    try {
        const int repetitions = parse_repetitions(argc, argv);
        std::mt19937 rng(0xB47C4u);

        // Small matrices make per-call overhead and batch scheduling relevant.
        constexpr int batch = 32;
        constexpr int m = 64;
        constexpr int n = 64;
        constexpr int k_size = 64;

        Matrix a(batch * matrix_size(m, k_size));
        Matrix b(batch * matrix_size(k_size, n));
        Matrix shared_b(matrix_size(k_size, n));
        Matrix expected(batch * matrix_size(m, n));
        Matrix actual(expected.size());
        fill_random(a, rng);
        fill_random(b, rng);
        fill_random(shared_b, rng);

        const auto clear_actual = [&] {
            std::fill(actual.begin(), actual.end(), 0.0f);
        };

        clear_actual();
        loop_of_gemms(a.data(), b.data(), actual.data(), batch, m, n, k_size);
        expected = actual;
        clear_actual();
        batch_matmul(a.data(), b.data(), actual.data(), batch, m, n, k_size);
        require_equal("BMM", expected, actual);

        const double looped_time = measure(
            clear_actual,
            [&] {
                loop_of_gemms(a.data(), b.data(), actual.data(), batch, m, n,
                              k_size);
            },
            actual, repetitions);
        const double bmm_time = measure(
            clear_actual,
            [&] {
                batch_matmul(a.data(), b.data(), actual.data(), batch, m, n,
                             k_size);
            },
            actual, repetitions);

        clear_actual();
        loop_with_shared_rhs(a.data(), shared_b.data(), actual.data(), batch, m,
                             n, k_size);
        expected = actual;
        clear_actual();
        flattened_shared_rhs(a.data(), shared_b.data(), actual.data(), batch, m,
                             n, k_size);
        require_equal("flattened shared-RHS GEMM", expected, actual);

        const double shared_loop_time = measure(
            clear_actual,
            [&] {
                loop_with_shared_rhs(a.data(), shared_b.data(), actual.data(),
                                     batch, m, n, k_size);
            },
            actual, repetitions);
        const double flattened_time = measure(
            clear_actual,
            [&] {
                flattened_shared_rhs(a.data(), shared_b.data(), actual.data(),
                                     batch, m, n, k_size);
            },
            actual, repetitions);

        std::cout << "Portable CPU structural study (not a vendor BLAS)\n\n"
                  << "1. Independent matrices\n"
                  << "   A[B,M,K] * B[B,K,N] -> C[B,M,N]\n"
                  << "   shape: batch=" << batch << ", M=N=K=" << m << "\n\n"
                  << std::fixed << std::setprecision(2)
                  << "   loop of GEMM calls: " << std::setw(10) << looped_time
                  << " us, " << std::setw(7)
                  << gemm_gflops(batch, m, n, k_size, looped_time)
                  << " GFLOP/s\n"
                  << "   one BMM call:       " << std::setw(10) << bmm_time
                  << " us, " << std::setw(7)
                  << gemm_gflops(batch, m, n, k_size, bmm_time)
                  << " GFLOP/s\n"
                  << "   loop/BMM:           " << looped_time / bmm_time
                  << "x\n\n"
                  << "2. Shared right-hand matrix\n"
                  << "   A[B,M,K] * shared_B[K,N]\n\n"
                  << "   loop of GEMM calls: " << std::setw(10)
                  << shared_loop_time << " us\n"
                  << "   reshape to GEMM:    " << std::setw(10)
                  << flattened_time << " us\n"
                  << "   loop/reshape:       "
                  << shared_loop_time / flattened_time << "x\n\n";

        // Decode-style attention: one query token per batch/head group.
        constexpr int model_batch = 2;
        constexpr int heads = 8;
        constexpr int groups = model_batch * heads;
        constexpr int query_tokens = 1;
        constexpr int context_tokens = 512;
        constexpr int head_dim = 64;

        Matrix query(groups * matrix_size(query_tokens, head_dim));
        Matrix key_cache(groups * matrix_size(context_tokens, head_dim));
        Matrix value_cache(groups * matrix_size(context_tokens, head_dim));
        Matrix scores(groups * matrix_size(query_tokens, context_tokens));
        Matrix attention_output(groups *
                                matrix_size(query_tokens, head_dim));
        Matrix expected_attention(attention_output.size());
        fill_random(query, rng);
        fill_random(key_cache, rng);
        fill_random(value_cache, rng);

        const auto clear_attention = [&] {
            std::fill(scores.begin(), scores.end(), 0.0f);
            std::fill(attention_output.begin(), attention_output.end(), 0.0f);
        };

        clear_attention();
        attention_looped(query.data(), key_cache.data(), value_cache.data(),
                         scores.data(), attention_output.data(), groups,
                         query_tokens, context_tokens, head_dim);
        expected_attention = attention_output;
        clear_attention();
        attention_batched(query.data(), key_cache.data(), value_cache.data(),
                          scores.data(), attention_output.data(), groups,
                          query_tokens, context_tokens, head_dim);
        require_equal("batched K/V attention", expected_attention,
                      attention_output);

        const double looped_attention_time = measure(
            clear_attention,
            [&] {
                attention_looped(
                    query.data(), key_cache.data(), value_cache.data(),
                    scores.data(), attention_output.data(), groups,
                    query_tokens, context_tokens, head_dim);
            },
            attention_output, repetitions);
        const double batched_attention_time = measure(
            clear_attention,
            [&] {
                attention_batched(
                    query.data(), key_cache.data(), value_cache.data(),
                    scores.data(), attention_output.data(), groups,
                    query_tokens, context_tokens, head_dim);
            },
            attention_output, repetitions);

        std::cout << "3. K/V-cache attention\n"
                  << "   groups = batch * heads = " << model_batch << " * "
                  << heads << " = " << groups << '\n'
                  << "   Q[groups,1,D] * K^T[groups,D,T] -> scores"
                     "[groups,1,T]\n"
                  << "   softmax(scores) * V[groups,T,D] -> output"
                     "[groups,1,D]\n"
                  << "   context=" << context_tokens
                  << ", head_dim=" << head_dim << "\n\n"
                  << "   loop over heads:    " << std::setw(10)
                  << looped_attention_time << " us\n"
                  << "   one batched call:   " << std::setw(10)
                  << batched_attention_time << " us\n"
                  << "   loop/batched:       "
                  << looped_attention_time / batched_attention_time << "x\n\n"
                  << "Interpretation: BMM changes scheduling, not the math or "
                     "FLOP count.\n"
                  << "Ratios near 1x are expected for this simple CPU kernel; "
                     "batched\n"
                  << "library/GPU kernels help most when many small products "
                     "need one dispatch.\n"
                  << "For decoding Q=1, both K/V products are batched "
                     "matrix-vector operations.\n";

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
