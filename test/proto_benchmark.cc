#include <benchmark/benchmark.h>

#include <random>

#include "proto.h"

std::vector<uint64_t> make_random(int log2_min, int log2_max) {
  std::mt19937 rng;
  std::uniform_int_distribution<uint64_t> dist(
      static_cast<uint64_t>(1) << log2_min, static_cast<uint64_t>(1) << log2_max);

  std::vector<uint64_t> values(32 * 1024);
  std::generate_n(&values[0], values.size(), [&]() { return dist(rng); });

  return values;
}

void BM_to_varint(benchmark::State& state) {
  auto values = make_random(state.range(0), state.range(1));
  while (state.KeepRunningBatch(values.size())) {
    for (uint64_t i : values) {
      benchmark::DoNotOptimize(proto::to_varint(i));
    }
  }
}

BENCHMARK(BM_to_varint)->ArgPair(0, 7)->ArgPair(8, 15)->ArgPair(16, 31)->ArgPair(32, 47);

void BM_write_varint(benchmark::State& state) {
  auto values = make_random(state.range(0), state.range(1));

  while (state.KeepRunningBatch(values.size())) {
    for (uint64_t i : values) {
      std::array<uint8_t, 12> dst;
      proto::write_varint(&dst[0], i);
      benchmark::DoNotOptimize(dst);
    }
  }
}

BENCHMARK(BM_write_varint)->ArgPair(0, 7)->ArgPair(8, 15)->ArgPair(16, 31)->ArgPair(32, 63);

void BM_write_padding(benchmark::State& state) {
  auto values = make_random(1, state.range(0));

  std::vector<uint8_t> dst(1 << state.range(0));
  while (state.KeepRunningBatch(values.size())) {
    for (uint64_t i : values) {
      proto::write_padding(&dst[0], 1, i);
      benchmark::DoNotOptimize(dst);
    }
  }
}

BENCHMARK(BM_write_padding)->DenseRange(5, 10);
