// Small CPU model of the physical KV-cache layout choices used by vLLM.
// This studies memory traversal only; it is not a vLLM GPU benchmark.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

using Clock = std::chrono::steady_clock;
using Buffer = std::vector<float>;

volatile float benchmark_sink = 0.0f;

struct CacheStrides {
    std::size_t block;
    std::size_t head;
    std::size_t token;
};

// Both layouts have the same logical [block, head, token, dim] coordinates.
// Only their physical strides differ.
CacheStrides hnd_strides(int heads, int block_size, int head_dim) {
    return {
        static_cast<std::size_t>(heads) * block_size * head_dim,
        static_cast<std::size_t>(block_size) * head_dim,
        static_cast<std::size_t>(head_dim),
    };
}

CacheStrides nhd_strides(int heads, int block_size, int head_dim) {
    return {
        static_cast<std::size_t>(block_size) * heads * head_dim,
        static_cast<std::size_t>(head_dim),
        static_cast<std::size_t>(heads) * head_dim,
    };
}

inline std::size_t cache_index(const CacheStrides &strides, int block,
                               int head, int token, int dim) {
    return static_cast<std::size_t>(block) * strides.block +
           static_cast<std::size_t>(head) * strides.head +
           static_cast<std::size_t>(token) * strides.token + dim;
}

// Put identical logical values into differently laid-out physical buffers.
void initialize_cache(Buffer &key, Buffer &value, const CacheStrides &strides,
                      int blocks, int heads, int block_size, int head_dim) {
    for (int block = 0; block < blocks; ++block) {
        for (int head = 0; head < heads; ++head) {
            for (int token = 0; token < block_size; ++token) {
                for (int dim = 0; dim < head_dim; ++dim) {
                    const std::size_t logical =
                        (((static_cast<std::size_t>(block) * heads + head) *
                              block_size +
                          token) *
                             head_dim +
                         dim);
                    const std::size_t physical =
                        cache_index(strides, block, head, token, dim);
                    key[physical] = static_cast<float>((logical * 17) % 1009) /
                                        1009.0f -
                                    0.5f;
                    value[physical] =
                        static_cast<float>((logical * 29 + 7) % 1013) /
                            1013.0f -
                        0.5f;
                }
            }
        }
    }
}

// One stride-aware attention-like memory kernel. The arithmetic and traversal
// are unchanged between HND and NHD; only the three strides differ.
NOINLINE void paged_attention_like(
    const Buffer &query, const Buffer &key_cache, const Buffer &value_cache,
    const std::vector<int> &block_table, Buffer &output,
    const CacheStrides &strides, int sequences, int heads,
    int logical_blocks_per_sequence, int block_size, int head_dim) {
    std::fill(output.begin(), output.end(), 0.0f);

    for (int sequence = 0; sequence < sequences; ++sequence) {
        for (int head = 0; head < heads; ++head) {
            const float *q =
                &query[(static_cast<std::size_t>(sequence) * heads + head) *
                       head_dim];
            float *out =
                &output[(static_cast<std::size_t>(sequence) * heads + head) *
                        head_dim];

            for (int logical_block = 0;
                 logical_block < logical_blocks_per_sequence;
                 ++logical_block) {
                const int physical_block =
                    block_table[sequence * logical_blocks_per_sequence +
                                logical_block];
                for (int token = 0; token < block_size; ++token) {
                    const std::size_t base =
                        cache_index(strides, physical_block, head, token, 0);
                    float score = 0.0f;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        score += q[dim] * key_cache[base + dim];
                    }
                    for (int dim = 0; dim < head_dim; ++dim) {
                        out[dim] += score * value_cache[base + dim];
                    }
                }
            }
        }
    }
}

// Cache insertion naturally starts with one token and writes all of its heads.
NOINLINE void write_tokens(Buffer &key_cache, Buffer &value_cache,
                           const Buffer &new_keys, const Buffer &new_values,
                           const std::vector<int> &slot_mapping,
                           const CacheStrides &strides, int heads,
                           int block_size, int head_dim) {
    for (std::size_t input_token = 0; input_token < slot_mapping.size();
         ++input_token) {
        const int slot = slot_mapping[input_token];
        const int block = slot / block_size;
        const int token = slot % block_size;
        for (int head = 0; head < heads; ++head) {
            const std::size_t source =
                (input_token * heads + head) * head_dim;
            const std::size_t destination =
                cache_index(strides, block, head, token, 0);
            std::memcpy(&key_cache[destination], &new_keys[source],
                        static_cast<std::size_t>(head_dim) * sizeof(float));
            std::memcpy(&value_cache[destination], &new_values[source],
                        static_cast<std::size_t>(head_dim) * sizeof(float));
        }
    }
}

float maximum_relative_error(const Buffer &expected, const Buffer &actual) {
    float maximum = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float scale = std::max(1.0f, std::abs(expected[i]));
        maximum =
            std::max(maximum, std::abs(expected[i] - actual[i]) / scale);
    }
    return maximum;
}

void require_equal(const char *name, const Buffer &expected,
                   const Buffer &actual) {
    const float error = maximum_relative_error(expected, actual);
    if (error > 1.0e-5f) {
        throw std::runtime_error(std::string(name) +
                                 " differs between layouts; error=" +
                                 std::to_string(error));
    }
}

template <typename Work>
double measure(Work work, const Buffer &observed, int repetitions) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    work();
    benchmark_sink = benchmark_sink + observed[observed.size() / 2];

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = Clock::now();
        work();
        const auto end = Clock::now();
        benchmark_sink = benchmark_sink + observed[observed.size() / 2];
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// Layer-major versus block-major transfer
// ---------------------------------------------------------------------------

NOINLINE void gather_blocks_from_layer_major(
    const Buffer &source, Buffer &destination,
    const std::vector<int> &selected_blocks, int layers, int total_blocks,
    int page_elements) {
    for (std::size_t selected = 0; selected < selected_blocks.size();
         ++selected) {
        const int block = selected_blocks[selected];
        for (int layer = 0; layer < layers; ++layer) {
            const std::size_t source_offset =
                (static_cast<std::size_t>(layer) * total_blocks + block) *
                page_elements;
            const std::size_t destination_offset =
                (selected * layers + layer) * page_elements;
            std::memcpy(&destination[destination_offset], &source[source_offset],
                        static_cast<std::size_t>(page_elements) * sizeof(float));
        }
    }
}

NOINLINE void gather_blocks_from_block_major(
    const Buffer &source, Buffer &destination,
    const std::vector<int> &selected_blocks, int layers, int page_elements) {
    const std::size_t block_elements =
        static_cast<std::size_t>(layers) * page_elements;
    for (std::size_t selected = 0; selected < selected_blocks.size();
         ++selected) {
        const std::size_t source_offset =
            static_cast<std::size_t>(selected_blocks[selected]) *
            block_elements;
        const std::size_t destination_offset = selected * block_elements;
        std::memcpy(&destination[destination_offset], &source[source_offset],
                    block_elements * sizeof(float));
    }
}

int parse_repetitions(int argc, char **argv) {
    if (argc > 2) {
        throw std::invalid_argument(std::string("Usage: ") + argv[0] +
                                    " [repetitions=9]");
    }
    if (argc == 1) return 9;
    const int repetitions = std::stoi(argv[1]);
    if (repetitions <= 0)
        throw std::invalid_argument("repetitions must be positive");
    return repetitions;
}

int main(int argc, char **argv) {
    try {
        const int repetitions = parse_repetitions(argc, argv);

        constexpr int physical_blocks = 512;
        constexpr int heads = 8;
        constexpr int block_size = 16;
        constexpr int head_dim = 64;
        constexpr int sequences = 32;
        constexpr int logical_blocks_per_sequence = 32;

        const std::size_t cache_elements =
            static_cast<std::size_t>(physical_blocks) * heads * block_size *
            head_dim;
        const CacheStrides hnd =
            hnd_strides(heads, block_size, head_dim);
        const CacheStrides nhd =
            nhd_strides(heads, block_size, head_dim);

        Buffer hnd_key(cache_elements), hnd_value(cache_elements);
        Buffer nhd_key(cache_elements), nhd_value(cache_elements);
        initialize_cache(hnd_key, hnd_value, hnd, physical_blocks, heads,
                         block_size, head_dim);
        initialize_cache(nhd_key, nhd_value, nhd, physical_blocks, heads,
                         block_size, head_dim);

        Buffer query(static_cast<std::size_t>(sequences) * heads * head_dim);
        for (std::size_t i = 0; i < query.size(); ++i) {
            query[i] = static_cast<float>((i * 11 + 3) % 257) / 257.0f - 0.5f;
        }

        std::vector<int> block_table(sequences *
                                     logical_blocks_per_sequence);
        for (int sequence = 0; sequence < sequences; ++sequence) {
            for (int logical_block = 0;
                 logical_block < logical_blocks_per_sequence;
                 ++logical_block) {
                block_table[sequence * logical_blocks_per_sequence +
                            logical_block] =
                    (sequence * 37 + logical_block * 13) % physical_blocks;
            }
        }

        Buffer hnd_output(static_cast<std::size_t>(sequences) * heads *
                          head_dim);
        Buffer nhd_output(hnd_output.size());
        const auto run_hnd_read = [&] {
            paged_attention_like(query, hnd_key, hnd_value, block_table,
                                 hnd_output, hnd, sequences, heads,
                                 logical_blocks_per_sequence, block_size,
                                 head_dim);
        };
        const auto run_nhd_read = [&] {
            paged_attention_like(query, nhd_key, nhd_value, block_table,
                                 nhd_output, nhd, sequences, heads,
                                 logical_blocks_per_sequence, block_size,
                                 head_dim);
        };
        run_hnd_read();
        run_nhd_read();
        require_equal("attention-like read", hnd_output, nhd_output);

        const double hnd_read_us =
            measure(run_hnd_read, hnd_output, repetitions);
        const double nhd_read_us =
            measure(run_nhd_read, nhd_output, repetitions);

        // Every cache slot is written exactly once, in a permuted order.
        const int inserted_tokens = physical_blocks * block_size;
        std::vector<int> slot_mapping(inserted_tokens);
        for (int token = 0; token < inserted_tokens; ++token) {
            slot_mapping[token] = (token * 8191) % inserted_tokens;
        }
        Buffer new_keys(static_cast<std::size_t>(inserted_tokens) * heads *
                        head_dim);
        Buffer new_values(new_keys.size());
        for (std::size_t i = 0; i < new_keys.size(); ++i) {
            new_keys[i] = static_cast<float>((i * 5 + 1) % 251) / 251.0f;
            new_values[i] = static_cast<float>((i * 7 + 2) % 241) / 241.0f;
        }
        const auto run_hnd_write = [&] {
            write_tokens(hnd_key, hnd_value, new_keys, new_values, slot_mapping,
                         hnd, heads, block_size, head_dim);
        };
        const auto run_nhd_write = [&] {
            write_tokens(nhd_key, nhd_value, new_keys, new_values, slot_mapping,
                         nhd, heads, block_size, head_dim);
        };
        const double hnd_write_us =
            measure(run_hnd_write, hnd_key, repetitions);
        const double nhd_write_us =
            measure(run_nhd_write, nhd_key, repetitions);

        std::cout << "Educational vLLM KV-layout study (CPU, not vLLM kernels)\n\n"
                  << "1. Inside each cache block/page\n"
                  << "   HND: [block, head, token, dim]\n"
                  << "   NHD: [block, token, head, dim]\n"
                  << "   Both use the same stride-aware kernel below.\n\n"
                  << std::fixed << std::setprecision(2)
                  << "   attention-like read HND: " << std::setw(10)
                  << hnd_read_us << " us\n"
                  << "   attention-like read NHD: " << std::setw(10)
                  << nhd_read_us << " us\n"
                  << "   NHD/HND read time:       " << nhd_read_us / hnd_read_us
                  << "x\n\n"
                  << "   cache write HND:         " << std::setw(10)
                  << hnd_write_us << " us\n"
                  << "   cache write NHD:         " << std::setw(10)
                  << nhd_write_us << " us\n"
                  << "   HND/NHD write time:      "
                  << hnd_write_us / nhd_write_us << "x\n\n";

        constexpr int layers = 16;
        constexpr int total_blocks = 128;
        constexpr int page_elements = 2048;
        constexpr int selected_count = 32;
        const std::size_t layered_elements =
            static_cast<std::size_t>(layers) * total_blocks * page_elements;
        Buffer layer_major(layered_elements);
        Buffer block_major(layered_elements);
        for (int layer = 0; layer < layers; ++layer) {
            for (int block = 0; block < total_blocks; ++block) {
                for (int element = 0; element < page_elements; ++element) {
                    const float value = static_cast<float>(
                                            ((layer * 131 + block * 17 + element) %
                                             1021)) /
                                        1021.0f;
                    layer_major[(static_cast<std::size_t>(layer) *
                                     total_blocks +
                                 block) *
                                    page_elements +
                                element] = value;
                    block_major[(static_cast<std::size_t>(block) * layers +
                                 layer) *
                                    page_elements +
                                element] = value;
                }
            }
        }
        std::vector<int> selected_blocks(selected_count);
        for (int i = 0; i < selected_count; ++i) {
            selected_blocks[i] = (i * 29) % total_blocks;
        }
        Buffer gathered_layer(static_cast<std::size_t>(selected_count) * layers *
                              page_elements);
        Buffer gathered_block(gathered_layer.size());
        const auto run_layer_transfer = [&] {
            gather_blocks_from_layer_major(layer_major, gathered_layer,
                                           selected_blocks, layers, total_blocks,
                                           page_elements);
        };
        const auto run_block_transfer = [&] {
            gather_blocks_from_block_major(block_major, gathered_block,
                                           selected_blocks, layers,
                                           page_elements);
        };
        run_layer_transfer();
        run_block_transfer();
        require_equal("cross-layer block transfer", gathered_layer,
                      gathered_block);
        const double layer_transfer_us =
            measure(run_layer_transfer, gathered_layer, repetitions);
        const double block_transfer_us =
            measure(run_block_transfer, gathered_block, repetitions);

        std::cout << "2. Across layers and blocks (an orthogonal choice)\n"
                  << "   LB...: [layer, block, page contents]\n"
                  << "   BL...: [block, layer, page contents]\n\n"
                  << "   gather blocks from LB:   " << std::setw(10)
                  << layer_transfer_us << " us\n"
                  << "   gather blocks from BL:   " << std::setw(10)
                  << block_transfer_us << " us\n"
                  << "   LB/BL transfer time:     "
                  << layer_transfer_us / block_transfer_us << "x\n\n"
                  << "Interpretation:\n"
                  << "- Layout changes addresses/strides, not attention math.\n"
                  << "- A stride-aware kernel can be the same kernel for HND and "
                     "NHD.\n"
                  << "- A hard-coded contiguous kernel needs another specialization "
                     "or conversion.\n"
                  << "- Block-major across layers mainly helps whole-block transfer; "
                     "it is\n"
                  << "  not the opposite of head-major inside a block.\n";

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
