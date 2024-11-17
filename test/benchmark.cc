#include <benchmark/benchmark.h>

#include <condition_variable>
#include <mutex>
#include <thread>

void BM_uncontended_mutex(benchmark::State& state) {
  std::mutex m;
  for (auto _ : state) {
    m.lock();
    m.unlock();
  }
}

BENCHMARK(BM_uncontended_mutex);

void BM_thread_pool(benchmark::State& state) {
  std::vector<std::thread> threads;
  std::mutex m;
  std::condition_variable cv;

  std::atomic<bool> run{true};

  for (int i = 0; i < state.range(0); ++i) {
    threads.emplace_back([&]() {
      std::unique_lock l(m);
      while (run) {
        cv.wait(l);
      }
    });
  }

  for (auto _ : state) {
    cv.notify_all();
  }
  run = false;
  cv.notify_all();
  for (auto& i : threads) {
    i.join();
  }
}

BENCHMARK(BM_thread_pool)->RangeMultiplier(2)->Range(2, 16);

void BM_multiple_locks(benchmark::State& state) {
  std::vector<std::thread> threads;
  std::mutex m0, m1, m2, m3;

  std::atomic<bool> run{true};

  for (int i = 0; i < state.range(0); ++i) {
    threads.emplace_back([&]() {
      int counter = 0;
      while (run) {
        if (counter++ & 1) {
          std::lock(m0, m1, m2, m3);
        } else {
          std::lock(m3, m2, m1, m0);
        }
        m0.unlock();
        m1.unlock();
        m2.unlock();
        m3.unlock();
      }
    });
  }

  for (auto _ : state) {
    std::unique_lock l(m0);
  }
  run = false;
  for (auto& i : threads) {
    i.join();
  }
}

BENCHMARK(BM_multiple_locks)->RangeMultiplier(2)->Range(2, 16);
