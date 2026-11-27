#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "paudit/inference.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using paudit::Metric;

TEST_CASE("Student-t tail against known values", "[inference]") {
  // scipy.stats.t.sf(t, df) * 2
  CHECK_THAT(paudit::student_t_two_sided_p(2.0, 10.0), WithinRel(0.07338803477074037, 1e-12));
  CHECK_THAT(paudit::student_t_two_sided_p(1.0, 1.0), WithinRel(0.5, 1e-12));
  CHECK_THAT(paudit::student_t_two_sided_p(0.0, 7.0), WithinAbs(1.0, 1e-15));
  CHECK(paudit::student_t_two_sided_p(1e300, 5.0) == 0.0);
  CHECK(std::isnan(paudit::student_t_two_sided_p(std::nan(""), 5.0)));
}

TEST_CASE("Diebold-Mariano: identical forecasts and unsupported metrics", "[inference]") {
  std::vector<double> y(50);
  std::vector<double> m(50);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = std::sin(static_cast<double>(i));
    m[i] = y[i] + 0.1;
  }
  const auto same = paudit::diebold_mariano(Metric::mae, y, m, m);
  CHECK(same.stat == 0.0);
  CHECK(same.p_value == 1.0);
  const auto r2 = paudit::diebold_mariano(Metric::r2, y, m, m);
  CHECK(std::isnan(r2.stat));
}

TEST_CASE("multiple-testing corrections", "[inference]") {
  const std::vector<double> p{0.001, 0.04, 0.03, 0.5, std::nan("")};
  const auto q = paudit::benjamini_hochberg(p);
  CHECK(std::isnan(q[4]));
  CHECK_THAT(q[0], WithinRel(0.004, 1e-12));
  CHECK_THAT(q[1], WithinRel(0.16 / 3.0, 1e-12));
  CHECK_THAT(q[2], WithinRel(0.16 / 3.0, 1e-12));
  CHECK_THAT(q[3], WithinRel(0.5, 1e-12));
  const auto h = paudit::holm_bonferroni(p);
  CHECK_THAT(h[0], WithinRel(0.004, 1e-12));
  CHECK_THAT(h[1], WithinRel(0.09, 1e-12));
  CHECK_THAT(h[2], WithinRel(0.09, 1e-12));
  CHECK_THAT(h[3], WithinRel(0.5, 1e-12));
}
