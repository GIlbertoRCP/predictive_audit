#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "paudit/numeric.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("pairwise sum follows NumPy's association order", "[numeric]") {
  // 1e16 + 1 + ... : naive left-to-right summation loses the ones, pairwise keeps some.
  std::vector<double> x(1000, 1.0);
  x[0] = 1e16;
  const double naive = [&] {
    double s = 0.0;
    for (const double v : x) s += v;
    return s;
  }();
  const double pw = paudit::numeric::sum(x);
  CHECK(naive == 1e16);
  CHECK(pw > naive);
  // Small inputs (< 8) are summed left to right starting from +0.0.
  CHECK(paudit::numeric::sum(std::vector<double>{1.0, 2.0, 3.0}) == 6.0);
  CHECK(std::isnan(paudit::numeric::mean(std::vector<double>{})));
}

TEST_CASE("multi-column pairwise sums equal single-column sums", "[numeric]") {
  std::vector<double> a(777);
  std::vector<double> b(777);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = std::sin(static_cast<double>(i)) * 1e3;
    b[i] = std::cos(static_cast<double>(i) * 0.7) / 3.0;
  }
  auto f = [&](std::size_t i, std::array<double, 2>& v) { v = {a[i], b[i]}; };
  std::array<double, 2> both{};
  paudit::numeric::pairwise_sum_k<2>(f, 0, a.size(), both);
  CHECK(both[0] == paudit::numeric::sum(a));
  CHECK(both[1] == paudit::numeric::sum(b));
}

TEST_CASE("percentile matches np.percentile linear interpolation", "[numeric]") {
  std::vector<double> v{5.0, 1.0, 4.0, 2.0, 3.0};
  CHECK(paudit::numeric::percentile_linear(v, 50.0) == 3.0);
  std::vector<double> w{5.0, 1.0, 4.0, 2.0, 3.0};
  CHECK_THAT(paudit::numeric::percentile_linear(w, 2.5), WithinAbs(1.1, 1e-15));
  std::vector<double> one{7.0};
  CHECK(paudit::numeric::percentile_linear(one, 97.5) == 7.0);
  std::vector<double> top{1.0, 2.0};
  CHECK(paudit::numeric::percentile_linear(top, 100.0) == 2.0);
}

TEST_CASE("prefix sums have a leading zero", "[numeric]") {
  const std::vector<double> x{1.0, 2.0, 3.0};
  std::vector<double> out(4);
  paudit::numeric::prefix_sums(x, out);
  CHECK(out == std::vector<double>{0.0, 1.0, 3.0, 6.0});
}
