// Study when block-major layout plus block-first traversal is useful.
//
// This is a CPU memory-locality model, not a vLLM GPU benchmark.  It keeps the
// logical cache contents and amount of useful work fixed while changing:
//   1. physical layout: [layer, block, page] or [block, layer, page]
//   2. loop order:      layer first or block first


#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
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

volatile double benchmark_sink = 0.0;

enum class Layout {
    layer_major,  // [layer, block, page]
    block_major,  // [block, layer, page]
};

enum class Traversal {
    layer_first,  // for layer: for block: page
    block_first,  // for block: for layer: page
};

std::size_t page_offset(Layout layout, int layer, int block, int layers,
                        int blocks, int page_elements) {
    if (layout == Layout::layer_major) {
        return (static_cast<std::size_t>(layer) * blocks + block) *
               page_elements;
    }
    return (static_cast<std::size_t>(block) * layers + layer) * page_elements;
}

void initialize(Buffer &cache, Layout layout, int layers, int blocks,
                int page_elements) {
    for (int layer = 0; layer < layers; ++layer) {
        for (int block = 0; block < blocks; ++block) {
            const std::size_t base =
                page_offset(layout, layer, block, layers, blocks,
                            page_elements);
            for (int element = 0; element < page_elements; ++element) {
                // Integer-valued floats make checksums independent of addition
                // order, so all four variants can be checked exactly.
                cache[base + element] =
                    static_cast<float>((layer * 11 + block * 5 + element) % 16);
            }
        }
    }
}

// Reading one float per 64-byte cache line makes this primarily a study of
// address order, prefetching and TLB/cache behavior rather than arithmetic.
NOINLINE double touch_page(const Buffer &cache, std::size_t base,
                           int page_elements) {
    constexpr int elements_per_cache_line = 64 / sizeof(float);
    double checksum = 0.0;
    for (int element = 0; element < page_elements;
         element += elements_per_cache_line) {
        checksum += cache[base + element];
    }
    return checksum;
}

NOINLINE double scan_selected_blocks(const Buffer &cache, Layout layout,
                                     Traversal traversal,
                                     const std::vector<int> &selected_blocks,
                                     int active_layers, int total_layers,
                                     int total_blocks, int page_elements) {
    double checksum = 0.0;
    if (traversal == Traversal::block_first) {
        for (const int block : selected_blocks) {
            for (int layer = 0; layer < active_layers; ++layer) {
                checksum += touch_page(
                    cache,
                    page_offset(layout, layer, block, total_layers,
                                total_blocks, page_elements),
                    page_elements);
            }
        }
    } else {
        for (int layer = 0; layer < active_layers; ++layer) {
            for (const int block : selected_blocks) {
                checksum += touch_page(
                    cache,
                    page_offset(layout, layer, block, total_layers,
                                total_blocks, page_elements),
                    page_elements);
            }
        }
    }
    return checksum;
}

NOINLINE double scan_selected_layers(const Buffer &cache, Layout layout,
                                     Traversal traversal,
                                     const std::vector<int> &selected_layers,
                                     int active_blocks, int total_layers,
                                     int total_blocks, int page_elements) {
    double checksum = 0.0;
    if (traversal == Traversal::layer_first) {
        for (const int layer : selected_layers) {
            for (int block = 0; block < active_blocks; ++block) {
                checksum += touch_page(
                    cache,
                    page_offset(layout, layer, block, total_layers,
                                total_blocks, page_elements),
                    page_elements);
            }
        }
    } else {
        for (int block = 0; block < active_blocks; ++block) {
            for (const int layer : selected_layers) {
                checksum += touch_page(
                    cache,
                    page_offset(layout, layer, block, total_layers,
                                total_blocks, page_elements),
                    page_elements);
            }
        }
    }
    return checksum;
}

void evict_caches(const Buffer &eviction_buffer) {
    double checksum = 0.0;
    constexpr std::size_t elements_per_cache_line = 64 / sizeof(float);
    for (std::size_t i = 0; i < eviction_buffer.size();
         i += elements_per_cache_line) {
        checksum += eviction_buffer[i];
    }
    benchmark_sink = benchmark_sink + checksum;
}

struct PairedTiming {
    double first_us;
    double second_us;
};

template <typename Work>
double timed_sample_us(Work work, const Buffer &eviction_buffer) {
    evict_caches(eviction_buffer);
    const auto start = Clock::now();
    const double checksum = work();
    const auto end = Clock::now();
    benchmark_sink = benchmark_sink + checksum;
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// Alternate which variant runs first. This reduces bias from frequency drift
// and from always measuring one layout earlier in a long benchmark run.
template <typename FirstWork, typename SecondWork>
PairedTiming measure_pair_us(FirstWork first, SecondWork second,
                             const Buffer &eviction_buffer, int repetitions) {
    std::vector<double> first_samples;
    std::vector<double> second_samples;
    first_samples.reserve(repetitions);
    second_samples.reserve(repetitions);

    // Prime both code paths and allow the CPU to leave an idle frequency state.
    // These samples are intentionally discarded.
    (void)timed_sample_us(first, eviction_buffer);
    (void)timed_sample_us(second, eviction_buffer);

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        if (repetition % 2 == 0) {
            first_samples.push_back(timed_sample_us(first, eviction_buffer));
            second_samples.push_back(timed_sample_us(second, eviction_buffer));
        } else {
            second_samples.push_back(timed_sample_us(second, eviction_buffer));
            first_samples.push_back(timed_sample_us(first, eviction_buffer));
        }
    }
    std::sort(first_samples.begin(), first_samples.end());
    std::sort(second_samples.begin(), second_samples.end());
    return {first_samples[first_samples.size() / 2],
            second_samples[second_samples.size() / 2]};
}

void require_same(double expected, double actual, const char *description) {
    if (expected != actual) {
        throw std::runtime_error(std::string(description) +
                                 " checksums differ");
    }
}

std::vector<int> dense_blocks(int count) {
    std::vector<int> result(count);
    for (int i = 0; i < count; ++i) result[i] = i;
    return result;
}

std::vector<int> sparse_blocks(int count, int total_blocks) {
    std::vector<int> result(count);
    // 73 is coprime with 256, so this visits unique, widely separated pages.
    for (int i = 0; i < count; ++i) result[i] = (i * 73) % total_blocks;
    return result;
}

int parse_repetitions(int argc, char **argv) {
    if (argc > 2) {
        throw std::invalid_argument(std::string("Usage: ") + argv[0] +
                                    " [repetitions=7]");
    }
    if (argc == 1) return 7;
    const int repetitions = std::stoi(argv[1]);
    if (repetitions <= 0) {
        throw std::invalid_argument("repetitions must be positive");
    }
    return repetitions;
}

int main(int argc, char **argv) {
    try {
        const int repetitions = parse_repetitions(argc, argv);

        constexpr int layers = 32;
        constexpr int blocks = 256;
        constexpr int page_elements = 1024;  // 4 KiB per layer/block page
        constexpr int selected_block_count = 64;
        constexpr int selected_layer_count = 8;

        const std::size_t cache_elements =
            static_cast<std::size_t>(layers) * blocks * page_elements;
        Buffer layer_major(cache_elements);
        Buffer block_major(cache_elements);
        initialize(layer_major, Layout::layer_major, layers, blocks,
                   page_elements);
        initialize(block_major, Layout::block_major, layers, blocks,
                   page_elements);

        // Larger than the data cache on typical laptop CPUs. It is touched
        // before every timed sample so a lucky warm-cache order does not decide
        // the result.
        Buffer eviction_buffer(16 * 1024 * 1024, 1.0f);  // 64 MiB

        const std::vector<int> every_block = dense_blocks(blocks);
        const std::vector<int> sparse = sparse_blocks(selected_block_count,
                                                      blocks);
        const std::vector<int> dense = dense_blocks(selected_block_count);

        const auto scan = [&](const Buffer &cache, Layout layout,
                              Traversal traversal,
                              const std::vector<int> &selected,
                              int active_layers) {
            return scan_selected_blocks(cache, layout, traversal, selected,
                                        active_layers, layers, blocks,
                                        page_elements);
        };

        // Dense full-cache traversal: the amount of useful work is identical.
        const double reference =
            scan(layer_major, Layout::layer_major, Traversal::layer_first,
                 every_block, layers);
        require_same(reference,
                     scan(layer_major, Layout::layer_major,
                          Traversal::block_first, every_block, layers),
                     "LB block-first");
        require_same(reference,
                     scan(block_major, Layout::block_major,
                          Traversal::layer_first, every_block, layers),
                     "BL layer-first");
        require_same(reference,
                     scan(block_major, Layout::block_major,
                          Traversal::block_first, every_block, layers),
                     "BL block-first");

        const PairedTiming matching_full = measure_pair_us(
            [&] {
                return scan(layer_major, Layout::layer_major,
                            Traversal::layer_first, every_block, layers);
            },
            [&] {
                return scan(block_major, Layout::block_major,
                            Traversal::block_first, every_block, layers);
            },
            eviction_buffer, repetitions);
        const double lb_lf = matching_full.first_us;
        const double bl_bf = matching_full.second_us;
        const PairedTiming mismatched_full = measure_pair_us(
            [&] {
                return scan(layer_major, Layout::layer_major,
                            Traversal::block_first, every_block, layers);
            },
            [&] {
                return scan(block_major, Layout::block_major,
                            Traversal::layer_first, every_block, layers);
            },
            eviction_buffer, repetitions);
        const double lb_bf = mismatched_full.first_us;
        const double bl_lf = mismatched_full.second_us;

        std::cout
            << "Block-major / block-first crossover study\n"
            << "CPU memory model; not a vLLM GPU benchmark\n"
            << "Cache: 32 layers x 256 blocks x 4 KiB pages = 32 MiB\n\n"
            << "1. Full dense traversal (microseconds)\n"
            << "   Rows are physical layouts; columns are loop orders.\n\n"
            << "                     layer-first   block-first\n"
            << std::fixed << std::setprecision(2)
            << "   layer-major " << std::setw(15) << lb_lf << std::setw(14)
            << lb_bf << '\n'
            << "   block-major " << std::setw(15) << bl_lf << std::setw(14)
            << bl_bf << "\n\n"
            << "   Matching the loop order to the physical layout makes the\n"
            << "   page stream contiguous. With every page visited, LB+LF and\n"
            << "   BL+BF should be similar.\n\n";

        std::cout
            << "2. Block-owned work: 64 sparse blocks, increasing layers/block\n"
            << "   Ratio > 1 means block-major + block-first is faster.\n\n"
            << "     layers   payload/block     LB+LF(us)     BL+BF(us)   "
               "LB/BL\n";
        for (const int active_layers : {1, 2, 4, 8, 16, 32}) {
            const double expected =
                scan(layer_major, Layout::layer_major,
                     Traversal::layer_first, sparse, active_layers);
            require_same(expected,
                         scan(block_major, Layout::block_major,
                              Traversal::block_first, sparse, active_layers),
                         "block-owned sweep");
            const PairedTiming timing = measure_pair_us(
                [&] {
                    return scan(layer_major, Layout::layer_major,
                                Traversal::layer_first, sparse, active_layers);
                },
                [&] {
                    return scan(block_major, Layout::block_major,
                                Traversal::block_first, sparse, active_layers);
                },
                eviction_buffer, repetitions);
            const double layer_time = timing.first_us;
            const double block_time = timing.second_us;
            const int payload_kib = active_layers * page_elements *
                                    static_cast<int>(sizeof(float)) / 1024;
            std::cout << std::setw(11) << active_layers << std::setw(14)
                      << (std::to_string(payload_kib) + " KiB")
                      << std::setw(15) << layer_time << std::setw(14)
                      << block_time << std::setw(9)
                      << layer_time / block_time << "x\n";
        }

        std::cout
            << "\n3. Does block selection density remove the advantage?\n"
            << "   All 32 layers are consumed. Ratio > 1 favors BL+BF.\n\n"
            << "       pattern     blocks     LB+LF(us)     BL+BF(us)   "
               "LB/BL\n";
        const auto report_pattern = [&](const char *name,
                                        const std::vector<int> &selected) {
            const double expected =
                scan(layer_major, Layout::layer_major,
                     Traversal::layer_first, selected, layers);
            require_same(expected,
                         scan(block_major, Layout::block_major,
                              Traversal::block_first, selected, layers),
                         name);
            const PairedTiming timing = measure_pair_us(
                [&] {
                    return scan(layer_major, Layout::layer_major,
                                Traversal::layer_first, selected, layers);
                },
                [&] {
                    return scan(block_major, Layout::block_major,
                                Traversal::block_first, selected, layers);
                },
                eviction_buffer, repetitions);
            const double layer_time = timing.first_us;
            const double block_time = timing.second_us;
            std::cout << std::setw(14) << name << std::setw(11)
                      << selected.size() << std::setw(15) << layer_time
                      << std::setw(14) << block_time << std::setw(9)
                      << layer_time / block_time << "x\n";
        };
        report_pattern("dense range", dense);
        report_pattern("sparse", sparse);
        report_pattern("all blocks", every_block);

        // Symmetric control: work owns layers rather than blocks.
        std::vector<int> selected_layers(selected_layer_count);
        for (int i = 0; i < selected_layer_count; ++i) {
            selected_layers[i] = (i * 5) % layers;
        }
        const auto scan_layers = [&](const Buffer &cache, Layout layout,
                                     Traversal traversal) {
            return scan_selected_layers(cache, layout, traversal,
                                        selected_layers, blocks, layers,
                                        blocks, page_elements);
        };
        const double layer_owned_reference =
            scan_layers(layer_major, Layout::layer_major,
                        Traversal::layer_first);
        require_same(layer_owned_reference,
                     scan_layers(block_major, Layout::block_major,
                                 Traversal::block_first),
                     "layer-owned control");
        const PairedTiming layer_owned = measure_pair_us(
            [&] {
                return scan_layers(layer_major, Layout::layer_major,
                                   Traversal::layer_first);
            },
            [&] {
                return scan_layers(block_major, Layout::block_major,
                                   Traversal::block_first);
            },
            eviction_buffer, repetitions);
        const double layer_owned_lb = layer_owned.first_us;
        const double layer_owned_bl = layer_owned.second_us;

        std::cout
            << "\n4. Counterexample: layer-owned work\n"
            << "   Eight sparse layers are read across all blocks.\n\n"
            << "   layer-major + layer-first: " << std::setw(10)
            << layer_owned_lb << " us\n"
            << "   block-major + block-first: " << std::setw(10)
            << layer_owned_bl << " us\n"
            << "   BL/LB time:                " << std::setw(10)
            << layer_owned_bl / layer_owned_lb << "x\n\n"
            << "Conclusion:\n"
            << "- BL+BF holds up when blocks are the unit of ownership or\n"
            << "  transfer, several layers are consumed per selected block,\n"
            << "  and many other blocks are skipped.\n"
            << "- Its advantage fades for one layer/block or a dense full-cache\n"
            << "  sweep; it reverses for layer-owned sparse work.\n";

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
