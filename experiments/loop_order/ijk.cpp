#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
typedef float vec __attribute__ (( vector_size(32) ));


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

    std::cout
        << std::setw(8)  << "n"
        << std::setw(14) << "Memory(MB)"
        << std::setw(18) << "Baseline(us)"
        << std::setw(18) << "Transpose(us)"
        << std::setw(14) << "Speedup"
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
        }

        // --------------------------------------------
        // Measurements
        // --------------------------------------------

        std::vector<double> baseline_times;
        std::vector<double> transposed_times;

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

            } else {

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

        double speedup =
            baseline / transposed;

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
            << std::setw(13) << speedup << "x"
            << "\n";
    }

    return 0;
}

// a helper function that allocates n vectors and initializes them with zeros
vec* alloc(int n) {
    vec* ptr = (vec*) std::aligned_alloc(32, 32 * n);
    memset(ptr, 0, 32 * n);
    return ptr;
}


void simd(vector<float> &A, vector<float> &B, vector<float> &C, int n) {
    // SIMD implementation would go here

    int Nb=n / 8; // Number of blocks of 8 floats
    vec *a=alloc(n*Nb);
    vec *b=alloc(n*Nb);

    //now these are vectors of vectors of 8 floats. 
    //now we will allocate A and B into these vectors of vectors of 8 floats.
    for(int i=0;i<n;i++){
        for(int j=0;j<Nb;j++){
            a[i*Nb+j/8][j%8]=A[i*n+j];
            b[i*Nb+j/8][j%8]=B[j*n+i];
        }
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            vec s{};
            for(int k=0;k<Nb;k++){
                s += a[i*Nb+k] * b[j*Nb+k];
            }
            for (int k = 0; k < 8; k++)
                C[i * n + j] += s[k];

        }
    }
}