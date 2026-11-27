#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "paudit/parallel.hpp"

TEST_CASE("parallel_for covers every index exactly once", "[parallel]") {
  for (const unsigned threads : {1U, 2U, 4U, 0U}) {
    paudit::parallel::set_num_threads(threads);
    std::vector<int> hits(10'007, 0);
    paudit::parallel::parallel_for(hits.size(), 97, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i) ++hits[i];
    });
    CHECK(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }));
  }
  paudit::parallel::set_num_threads(0);
}

TEST_CASE("nested parallel_for runs inline", "[parallel]") {
  paudit::parallel::set_num_threads(4);
  std::atomic<long> total{0};
  paudit::parallel::parallel_for(8, 1, [&](std::size_t, std::size_t) {
    paudit::parallel::parallel_for(100, 10,
                                   [&](std::size_t b, std::size_t e) { total += static_cast<long>(e - b); });
  });
  CHECK(total == 800);
  paudit::parallel::set_num_threads(0);
}

TEST_CASE("exceptions propagate to the caller", "[parallel]") {
  paudit::parallel::set_num_threads(4);
  CHECK_THROWS_AS(paudit::parallel::parallel_for(1000, 10,
                                                 [](std::size_t b, std::size_t) {
                                                   if (b == 500) throw std::runtime_error("boom");
                                                 }),
                  std::runtime_error);
  // The pool is still usable afterwards.
  std::atomic<int> n{0};
  paudit::parallel::parallel_for(100, 10,
                                 [&](std::size_t b, std::size_t e) { n += static_cast<int>(e - b); });
  CHECK(n == 100);
  paudit::parallel::set_num_threads(0);
}

TEST_CASE("changing the thread count while a job runs is safe", "[parallel]") {
  paudit::parallel::set_num_threads(4);
  std::atomic<bool> stop{false};
  std::atomic<long> work{0};
  std::thread runner([&] {
    while (!stop) {
      paudit::parallel::parallel_for(4096, 16,
                                     [&](std::size_t b, std::size_t e) { work += static_cast<long>(e - b); });
    }
  });
  for (unsigned n : {2U, 8U, 1U, 3U, 0U}) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    paudit::parallel::set_num_threads(n);
  }
  stop = true;
  runner.join();
  CHECK(work > 0);
  CHECK(work % 4096 == 0);  // every job covered its full range exactly once
  paudit::parallel::set_num_threads(0);
}
