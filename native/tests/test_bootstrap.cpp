#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <span>
#include <vector>

#include "paudit/bootstrap.hpp"
#include "paudit/parallel.hpp"

using paudit::Metric;

namespace {
std::vector<double> wave(std::size_t n, double phase, double noise) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto t = static_cast<double>(i);
    v[i] = std::sin(t * 0.1 + phase) * 3.0 + noise * std::sin(t * 7.3);
  }
  return v;
}
}  // namespace

TEST_CASE("circular layout never crosses group boundaries", "[bootstrap]") {
  std::vector<std::int64_t> groups;
  for (std::int64_t g = 0; g < 5; ++g) groups.insert(groups.end(), 15, g);
  const auto layout = paudit::group_circular_layout(groups, 9);
  CHECK(layout.starts.size() == groups.size());
  for (const auto s : layout.starts) {
    const auto g0 = groups[static_cast<std::size_t>(layout.gather[static_cast<std::size_t>(s)])];
    for (std::size_t j = 0; j < 10; ++j) {
      CHECK(groups[static_cast<std::size_t>(layout.gather[static_cast<std::size_t>(s) + j])] == g0);
    }
  }
}

TEST_CASE("aligned panels are detected from integer and float times", "[bootstrap]") {
  std::vector<std::int64_t> groups;
  std::vector<std::int64_t> ti;
  std::vector<double> tf;
  for (std::int64_t g = 0; g < 4; ++g) {
    for (std::int64_t t = 0; t < 30; ++t) {
      groups.push_back(g);
      ti.push_back(t + 100);
      tf.push_back(static_cast<double>(t) * 0.5);
    }
  }
  const std::span<const std::int64_t> gs(groups);
  const auto a = paudit::detect_aligned_panel(gs, std::span<const std::int64_t>(ti));
  CHECK(a.aligned);
  CHECK(a.n_times == 30);
  CHECK(a.time_index[31] == 1);
  const auto b = paudit::detect_aligned_panel(gs, std::span<const double>(tf));
  CHECK(b.aligned);
  CHECK(b.n_times == 30);
  const auto c = paudit::detect_aligned_panel(std::nullopt, std::span<const double>(tf));
  CHECK_FALSE(c.aligned);
  CHECK(paudit::default_block_length(groups.size(), gs, std::span<const std::int64_t>(ti)) == 10);
}

TEST_CASE("bootstrap is independent of the thread count", "[bootstrap]") {
  const std::size_t n = 5000;
  const auto y = wave(n, 0.0, 0.0);
  const auto m = wave(n, 0.05, 0.2);
  const auto b = wave(n, 0.3, 0.4);
  paudit::parallel::set_num_threads(1);
  const auto one = paudit::block_bootstrap_diff(Metric::mae, y, m, b, 23, 300, 5);
  paudit::parallel::set_num_threads(8);
  const auto many = paudit::block_bootstrap_diff(Metric::mae, y, m, b, 23, 300, 5);
  paudit::parallel::set_num_threads(0);
  CHECK(one.lo == many.lo);
  CHECK(one.hi == many.hi);
  CHECK(one.lo < one.hi);
  CHECK(one.lo > 0.0);  // the model is clearly closer to y than the baseline
}

TEST_CASE("degenerate inputs return NaN intervals", "[bootstrap]") {
  const std::vector<double> empty;
  const auto r = paudit::block_bootstrap_diff(Metric::mae, empty, empty, empty, 5, 100, 0);
  CHECK(std::isnan(r.lo));
  const std::vector<double> y{1, 2, 3};
  const auto z = paudit::block_bootstrap_diff(Metric::mae, y, y, y, 5, 0, 0);
  CHECK(std::isnan(z.hi));
  CHECK_THROWS_AS(paudit::block_bootstrap_diff(Metric::roc_auc, y, y, y, 1, 10, 0), std::invalid_argument);
}

TEST_CASE("Politis-White block length grows with autocorrelation", "[bootstrap]") {
  std::vector<double> iid(2000);
  std::vector<double> ar(2000);
  double state = 0.0;
  for (std::size_t i = 0; i < iid.size(); ++i) {
    const double e = std::sin(static_cast<double>(i) * 12.9898) * 43758.5453;
    const double u = e - std::floor(e) - 0.5;
    iid[i] = u;
    state = 0.95 * state + u;
    ar[i] = state;
  }
  CHECK(paudit::estimate_block_length(iid) < paudit::estimate_block_length(ar));
  CHECK(paudit::estimate_block_length(std::vector<double>(100, 0.0)) >= 1);
  CHECK(paudit::estimate_block_length(std::vector<double>{1.0, 2.0, 3.0}) >= 1);
}
