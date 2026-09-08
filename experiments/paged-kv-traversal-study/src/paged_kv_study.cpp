#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

enum class Layout { BHND, HBND, BNHD };
enum class Stage { Fixed, Matched, Full };
enum class BlockOrder { Sequential, Shuffled, Fragmented };

struct Config {
  std::size_t blocks = 128;
  std::size_t heads = 16;
  std::size_t block_size = 16;
  std::size_t head_dim = 64;
  int warmups = 1;
  int repeats = 5;
  Stage stage = Stage::Fixed;
  BlockOrder block_order = BlockOrder::Sequential;
  std::size_t block_run_length = 1;
  std::uint32_t block_seed = 0;
  bool scrub_cache = true;
  std::optional<Layout> selected_memory_layout;
  std::optional<Layout> selected_traversal;
  std::string csv_path;
};

struct Storage {
  std::vector<float> key;
  std::vector<float> value;
};

struct Result {
  Layout memory_layout;
  Layout traversal;
  double median_ms;
  double gib_per_second;
  double gflops;
  double useful_flops_per_kv_byte;
  double checksum;
  double max_abs_error;
};

volatile double benchmark_sink = 0.0;

constexpr std::string_view name(Layout layout) {
  switch (layout) {
    case Layout::BHND:
      return "BHND";
    case Layout::HBND:
      return "HBND";
    case Layout::BNHD:
      return "BNHD";
  }
  return "unknown";
}

std::size_t checked_elements(const Config& config) {
  const long double count = static_cast<long double>(config.blocks) *
                            static_cast<long double>(config.heads) *
                            static_cast<long double>(config.block_size) *
                            static_cast<long double>(config.head_dim);
  if (count > static_cast<long double>(SIZE_MAX)) {
    throw std::overflow_error("tensor dimensions overflow size_t");
  }
  return config.blocks * config.heads * config.block_size * config.head_dim;
}

template <Layout memory_layout>
inline std::size_t offset(const Config& c, std::size_t block,
                          std::size_t head, std::size_t token,
                          std::size_t dim) {
  if constexpr (memory_layout == Layout::BHND) {
    return (((block * c.heads + head) * c.block_size + token) * c.head_dim +
            dim);
  } else if constexpr (memory_layout == Layout::HBND) {
    return (((head * c.blocks + block) * c.block_size + token) * c.head_dim +
            dim);
  } else {
    return (((block * c.block_size + token) * c.heads + head) * c.head_dim +
            dim);
  }
}

float deterministic_value(std::size_t block, std::size_t head,
                          std::size_t token, std::size_t dim,
                          std::uint32_t salt) {
  std::uint32_t x = static_cast<std::uint32_t>(block * 73856093ULL) ^
                    static_cast<std::uint32_t>(head * 19349663ULL) ^
                    static_cast<std::uint32_t>(token * 83492791ULL) ^
                    static_cast<std::uint32_t>(dim * 2654435761ULL) ^ salt;
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  return static_cast<float>(x & 0x3ffU) / 1024.0F - 0.5F;
}

template <Layout memory_layout>
Storage make_storage(const Config& c) {
  Storage storage{std::vector<float>(checked_elements(c)),
                  std::vector<float>(checked_elements(c))};
  for (std::size_t b = 0; b < c.blocks; ++b) {
    for (std::size_t h = 0; h < c.heads; ++h) {
      for (std::size_t n = 0; n < c.block_size; ++n) {
        for (std::size_t d = 0; d < c.head_dim; ++d) {
          const auto index = offset<memory_layout>(c, b, h, n, d);
          storage.key[index] = deterministic_value(b, h, n, d, 0x12345678U);
          storage.value[index] = deterministic_value(b, h, n, d, 0x9abcdef0U);
        }
      }
    }
  }
  return storage;
}

std::vector<float> make_query(const Config& c) {
  std::vector<float> query(c.heads * c.head_dim);
  for (std::size_t h = 0; h < c.heads; ++h) {
    for (std::size_t d = 0; d < c.head_dim; ++d) {
      query[h * c.head_dim + d] =
          deterministic_value(0, h, 0, d, 0x13579bdfU);
    }
  }
  return query;
}

std::vector<std::size_t> make_block_table(const Config& c) {
  std::vector<std::size_t> block_table(c.blocks);
  std::iota(block_table.begin(), block_table.end(), 0);
  if (c.block_order == BlockOrder::Shuffled) {
    std::mt19937 generator(c.block_seed);
    std::shuffle(block_table.begin(), block_table.end(), generator);
  } else if (c.block_order == BlockOrder::Fragmented) {
    std::vector<std::size_t> run_starts;
    for (std::size_t start = 0; start < c.blocks;
         start += c.block_run_length) {
      run_starts.push_back(start);
    }
    std::mt19937 generator(c.block_seed);
    std::shuffle(run_starts.begin(), run_starts.end(), generator);

    block_table.clear();
    block_table.reserve(c.blocks);
    for (const std::size_t start : run_starts) {
      const std::size_t stop = std::min(start + c.block_run_length, c.blocks);
      for (std::size_t block = start; block < stop; ++block) {
        block_table.push_back(block);
      }
    }
  }
  return block_table;
}

template <Layout memory_layout>
inline void visit_token(const Config& c, const Storage& storage,
                        const std::vector<float>& query,
                        std::vector<float>& output,
                        std::vector<float>& running_max,
                        std::vector<float>& running_sum, std::size_t block,
                        std::size_t head, std::size_t token) {
  const auto base = offset<memory_layout>(c, block, head, token, 0);
  const auto query_base = head * c.head_dim;
  float score = 0.0F;
  for (std::size_t d = 0; d < c.head_dim; ++d) {
    score += query[query_base + d] * storage.key[base + d];
  }
  score /= std::sqrt(static_cast<float>(c.head_dim));

  const float new_max = std::max(running_max[head], score);
  const float previous_weight =
      std::isinf(running_max[head]) ? 0.0F
                                    : std::exp(running_max[head] - new_max);
  const float current_weight = std::exp(score - new_max);
  for (std::size_t d = 0; d < c.head_dim; ++d) {
    output[query_base + d] = output[query_base + d] * previous_weight +
                             current_weight * storage.value[base + d];
  }
  running_sum[head] = running_sum[head] * previous_weight + current_weight;
  running_max[head] = new_max;
}

template <Layout memory_layout, Layout traversal>
void run_kernel(const Config& c, const Storage& storage,
                const std::vector<float>& query,
                const std::vector<std::size_t>& block_table,
                std::vector<float>& output, std::vector<float>& running_max,
                std::vector<float>& running_sum) {
  if constexpr (traversal == Layout::BHND) {
    for (std::size_t logical_block = 0; logical_block < c.blocks;
         ++logical_block) {
      const std::size_t b = block_table[logical_block];
      for (std::size_t h = 0; h < c.heads; ++h) {
        for (std::size_t n = 0; n < c.block_size; ++n) {
          visit_token<memory_layout>(c, storage, query, output, running_max,
                                     running_sum, b, h, n);
        }
      }
    }
  } else if constexpr (traversal == Layout::HBND) {
    for (std::size_t h = 0; h < c.heads; ++h) {
      for (std::size_t logical_block = 0; logical_block < c.blocks;
           ++logical_block) {
        const std::size_t b = block_table[logical_block];
        for (std::size_t n = 0; n < c.block_size; ++n) {
          visit_token<memory_layout>(c, storage, query, output, running_max,
                                     running_sum, b, h, n);
        }
      }
    }
  } else {
    for (std::size_t logical_block = 0; logical_block < c.blocks;
         ++logical_block) {
      const std::size_t b = block_table[logical_block];
      for (std::size_t n = 0; n < c.block_size; ++n) {
        for (std::size_t h = 0; h < c.heads; ++h) {
          visit_token<memory_layout>(c, storage, query, output, running_max,
                                     running_sum, b, h, n);
        }
      }
    }
  }

  for (std::size_t h = 0; h < c.heads; ++h) {
    const auto output_base = h * c.head_dim;
    for (std::size_t d = 0; d < c.head_dim; ++d) {
      output[output_base + d] /= running_sum[h];
    }
  }
}

void scrub_cache(std::vector<float>& scrubber) {
  double sum = 0.0;
  for (std::size_t i = 0; i < scrubber.size(); i += 16) {
    scrubber[i] += 1.0F;
    sum += scrubber[i];
  }
  benchmark_sink = sum;
}

double checksum(const std::vector<float>& output) {
  return std::accumulate(output.begin(), output.end(), 0.0);
}

double max_abs_difference(const std::vector<float>& lhs,
                          const std::vector<float>& rhs) {
  double maximum = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    maximum = std::max(maximum,
                       std::abs(static_cast<double>(lhs[i]) - rhs[i]));
  }
  return maximum;
}

template <Layout memory_layout, Layout traversal>
Result benchmark(const Config& c, const Storage& storage,
                 const std::vector<float>& query,
                 const std::vector<std::size_t>& block_table,
                 std::vector<float>& last_output,
                 std::vector<float>& scrubber) {
  std::vector<float> output(c.heads * c.head_dim);
  std::vector<float> running_max(c.heads);
  std::vector<float> running_sum(c.heads);

  for (int i = 0; i < c.warmups; ++i) {
    std::fill(output.begin(), output.end(), 0.0F);
    std::fill(running_max.begin(), running_max.end(),
              -std::numeric_limits<float>::infinity());
    std::fill(running_sum.begin(), running_sum.end(), 0.0F);
    run_kernel<memory_layout, traversal>(c, storage, query, block_table, output,
                                         running_max, running_sum);
    benchmark_sink = checksum(output);
  }

  std::vector<double> milliseconds;
  milliseconds.reserve(static_cast<std::size_t>(c.repeats));
  for (int i = 0; i < c.repeats; ++i) {
    if (c.scrub_cache) {
      scrub_cache(scrubber);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    std::fill(running_max.begin(), running_max.end(),
              -std::numeric_limits<float>::infinity());
    std::fill(running_sum.begin(), running_sum.end(), 0.0F);
    const auto start = std::chrono::steady_clock::now();
    run_kernel<memory_layout, traversal>(c, storage, query, block_table, output,
                                         running_max, running_sum);
    const auto stop = std::chrono::steady_clock::now();
    milliseconds.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
    benchmark_sink = checksum(output);
  }

  std::sort(milliseconds.begin(), milliseconds.end());
  const double median_ms = milliseconds[milliseconds.size() / 2];
  const double seconds = median_ms / 1000.0;
  const double elements = static_cast<double>(checked_elements(c));
  const double bytes = 2.0 * elements * sizeof(float);
  const double useful_flops = 4.0 * elements;
  last_output = output;
  return Result{memory_layout,
                traversal,
                median_ms,
                bytes / seconds / static_cast<double>(1ULL << 30),
                useful_flops / seconds / 1.0e9,
                useful_flops / bytes,
                checksum(output),
                0.0};
}

template <Layout memory_layout>
void benchmark_layout(const Config& c, const std::vector<Layout>& traversals,
                      const std::vector<float>& query,
                      const std::vector<std::size_t>& block_table,
                      std::vector<float>& reference,
                      std::vector<Result>& results,
                      std::vector<float>& scrubber) {
  auto storage = make_storage<memory_layout>(c);
  for (const Layout traversal : traversals) {
    std::vector<float> output;
    Result result{};
    if (traversal == Layout::BHND) {
      result = benchmark<memory_layout, Layout::BHND>(
          c, storage, query, block_table, output, scrubber);
    } else if (traversal == Layout::HBND) {
      result = benchmark<memory_layout, Layout::HBND>(
          c, storage, query, block_table, output, scrubber);
    } else {
      result = benchmark<memory_layout, Layout::BNHD>(
          c, storage, query, block_table, output, scrubber);
    }

    if (reference.empty()) {
      reference = output;
    }
    result.max_abs_error = max_abs_difference(reference, output);
    results.push_back(result);
  }
}

void benchmark_selected_layout(
    const Config& c, Layout memory_layout, Layout traversal,
    const std::vector<float>& query,
    const std::vector<std::size_t>& block_table, std::vector<float>& reference,
    std::vector<Result>& results, std::vector<float>& scrubber) {
  const std::vector<Layout> traversals = {traversal};
  if (memory_layout == Layout::BHND) {
    benchmark_layout<Layout::BHND>(c, traversals, query, block_table, reference,
                                   results, scrubber);
  } else if (memory_layout == Layout::HBND) {
    benchmark_layout<Layout::HBND>(c, traversals, query, block_table, reference,
                                   results, scrubber);
  } else {
    benchmark_layout<Layout::BNHD>(c, traversals, query, block_table, reference,
                                   results, scrubber);
  }
}

std::size_t parse_size(const std::string& value, std::string_view option) {
  const auto parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(std::string(option) + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

Layout parse_layout(const std::string& value, std::string_view option) {
  if (value == "BHND") {
    return Layout::BHND;
  }
  if (value == "HBND") {
    return Layout::HBND;
  }
  if (value == "BNHD") {
    return Layout::BNHD;
  }
  throw std::invalid_argument(std::string(option) + " must be BHND, HBND, or BNHD");
}

void print_help(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "  --stage fixed|matched|full\n"
      << "                       fixed: default BNHD/NHD memory\n"
      << "                       matched: three memory-matched cases\n"
      << "                       full: all 3 memory x 3 traversal combinations\n"
      << "  --memory-layout L    run one physical layout (requires --traversal)\n"
      << "  --traversal L        run one traversal (requires --memory-layout)\n"
      << "  --block-order sequential|shuffled|fragmented\n"
      << "  --block-run-length N contiguous blocks per fragmented run\n"
      << "  --block-seed N       deterministic permutation seed (default 0)\n"
      << "  --blocks N           physical blocks (default 128)\n"
      << "  --heads N            KV heads (default 16)\n"
      << "  --block-size N       tokens per block (default 16)\n"
      << "  --head-dim N         values per K/V head (default 64)\n"
      << "  --warmups N          untimed runs per case (default 1)\n"
      << "  --repeats N          timed runs per case (default 5)\n"
      << "  --no-cache-scrub     omit cache scrub before timed repetitions\n"
      << "  --csv PATH           also write machine-readable results\n";
}

Config parse_args(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help") {
      print_help(argv[0]);
      std::exit(0);
    }
    if (option == "--no-cache-scrub") {
      config.scrub_cache = false;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const std::string value = argv[++i];
    if (option == "--stage") {
      if (value == "fixed") {
        config.stage = Stage::Fixed;
      } else if (value == "matched") {
        config.stage = Stage::Matched;
      } else if (value == "full") {
        config.stage = Stage::Full;
      } else {
        throw std::invalid_argument("--stage must be fixed, matched, or full");
      }
    } else if (option == "--memory-layout") {
      config.selected_memory_layout = parse_layout(value, option);
    } else if (option == "--traversal") {
      config.selected_traversal = parse_layout(value, option);
    } else if (option == "--blocks") {
      config.blocks = parse_size(value, option);
    } else if (option == "--heads") {
      config.heads = parse_size(value, option);
    } else if (option == "--block-size") {
      config.block_size = parse_size(value, option);
    } else if (option == "--head-dim") {
      config.head_dim = parse_size(value, option);
    } else if (option == "--warmups") {
      config.warmups = static_cast<int>(parse_size(value, option));
    } else if (option == "--repeats") {
      config.repeats = static_cast<int>(parse_size(value, option));
    } else if (option == "--block-order") {
      if (value == "sequential") {
        config.block_order = BlockOrder::Sequential;
      } else if (value == "shuffled") {
        config.block_order = BlockOrder::Shuffled;
      } else if (value == "fragmented") {
        config.block_order = BlockOrder::Fragmented;
      } else {
        throw std::invalid_argument(
            "--block-order must be sequential, shuffled, or fragmented");
      }
    } else if (option == "--block-run-length") {
      config.block_run_length = parse_size(value, option);
    } else if (option == "--block-seed") {
      const auto seed = std::stoull(value);
      if (seed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--block-seed exceeds uint32 range");
      }
      config.block_seed = static_cast<std::uint32_t>(seed);
    } else if (option == "--csv") {
      config.csv_path = value;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (config.selected_memory_layout.has_value() !=
      config.selected_traversal.has_value()) {
    throw std::invalid_argument(
        "--memory-layout and --traversal must be specified together");
  }
  return config;
}

void write_csv(const std::string& path, const std::vector<Result>& results) {
  if (path.empty()) {
    return;
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("could not open CSV file: " + path);
  }
  output << "memory_layout,traversal,median_ms,gib_per_second,gflops,"
            "useful_flops_per_kv_byte,checksum,max_abs_error\n";
  output << std::setprecision(10);
  for (const auto& result : results) {
    output << name(result.memory_layout) << ',' << name(result.traversal) << ','
           << result.median_ms << ',' << result.gib_per_second << ','
           << result.gflops << ',' << result.useful_flops_per_kv_byte << ','
           << result.checksum << ','
           << result.max_abs_error << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config config = parse_args(argc, argv);
    const double kib = static_cast<double>(checked_elements(config)) *
                       2.0 * sizeof(float) / 1024.0;
    std::cout << "Paged KV traversal study (B means physical block, not batch)\n"
              << "Shape: B=" << config.blocks << " H=" << config.heads
              << " N=" << config.block_size << " D=" << config.head_dim
              << " | K+V=" << std::fixed << std::setprecision(2)
              << kib / 1024.0 << " MiB\n"
              << "Kernel: decode attention with online softmax over paged K/V\n"
              << "Useful arithmetic intensity: 0.500 FLOP/KV byte "
                 "(QK + weighted-V)\n"
              << "Block table: ";
    if (config.block_order == BlockOrder::Sequential) {
      std::cout << "sequential";
    } else if (config.block_order == BlockOrder::Shuffled) {
      std::cout << "shuffled (seed " << config.block_seed << ')';
    } else {
      std::cout << "fragmented (contiguous run " << config.block_run_length
                << ", seed " << config.block_seed << ')';
    }
    std::cout << "\nStage: ";
    if (config.selected_memory_layout) {
      std::cout << "single case";
    } else if (config.stage == Stage::Fixed) {
      std::cout << "fixed BNHD/NHD memory";
    } else if (config.stage == Stage::Matched) {
      std::cout << "three memory-matched cases";
    } else {
      std::cout << "full 3x3";
    }
    std::cout << "\n\n";

    const auto query = make_query(config);
    const auto block_table = make_block_table(config);
    std::vector<float> scrubber(16ULL * 1024ULL * 1024ULL, 1.0F);
    std::vector<float> reference;
    std::vector<Result> results;

    if (config.selected_memory_layout) {
      benchmark_selected_layout(
          config, *config.selected_memory_layout, *config.selected_traversal,
          query, block_table, reference, results, scrubber);
    } else if (config.stage == Stage::Fixed) {
      benchmark_layout<Layout::BNHD>(
          config, {Layout::BHND, Layout::HBND, Layout::BNHD}, query,
          block_table, reference, results, scrubber);
    } else if (config.stage == Stage::Matched) {
      benchmark_layout<Layout::BHND>(config, {Layout::BHND}, query, block_table,
                                     reference, results, scrubber);
      benchmark_layout<Layout::HBND>(config, {Layout::HBND}, query, block_table,
                                     reference, results, scrubber);
      benchmark_layout<Layout::BNHD>(config, {Layout::BNHD}, query, block_table,
                                     reference, results, scrubber);
    } else {
      const std::vector<Layout> traversals = {Layout::BHND, Layout::HBND,
                                               Layout::BNHD};
      benchmark_layout<Layout::BHND>(config, traversals, query, block_table,
                                     reference, results, scrubber);
      benchmark_layout<Layout::HBND>(config, traversals, query, block_table,
                                     reference, results, scrubber);
      benchmark_layout<Layout::BNHD>(config, traversals, query, block_table,
                                     reference, results, scrubber);
    }

    std::cout << std::left << std::setw(12) << "memory" << std::setw(12)
              << "traversal" << std::right << std::setw(13) << "median(ms)"
              << std::setw(13) << "GiB/s" << std::setw(13) << "GFLOP/s"
              << std::setw(15) << "max error" << '\n';
    for (const auto& result : results) {
      std::cout << std::left << std::setw(12) << name(result.memory_layout)
                << std::setw(12) << name(result.traversal) << std::right
                << std::fixed << std::setprecision(3) << std::setw(13)
                << result.median_ms << std::setw(13) << result.gib_per_second
                << std::setw(13) << result.gflops << std::scientific
                << std::setprecision(2) << std::setw(15)
                << result.max_abs_error << '\n';
    }

    const double max_reference = std::transform_reduce(
        reference.begin(), reference.end(), 0.0, [](double a, double b) {
          return std::max(a, b);
        }, [](float value) { return std::abs(static_cast<double>(value)); });
    const double tolerance = 1.0e-4 * (1.0 + max_reference);
    const bool correct = std::all_of(results.begin(), results.end(),
                                     [tolerance](const Result& result) {
                                       return result.max_abs_error <= tolerance;
                                     });
    std::cout << "\nCorrectness: " << (correct ? "PASS" : "FAIL")
              << " (tolerance " << std::scientific << tolerance << ")\n";
    write_csv(config.csv_path, results);
    return correct ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
