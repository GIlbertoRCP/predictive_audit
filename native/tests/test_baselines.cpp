#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include "paudit/baselines.hpp"

using Catch::Matchers::WithinAbs;

namespace {
const double kNaN = std::nan("");
bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }
}  // namespace

TEST_CASE("groups keep first-appearance order and original row order", "[baselines]") {
  const std::vector<std::int64_t> g{7, 3, 7, 3, 9};
  const auto rows = paudit::group_rows(g.size(), std::span<const std::int64_t>(g));
  CHECK(rows.n_groups() == 3);
  CHECK(rows.rows == std::vector<std::size_t>{0, 2, 1, 3, 4});
  CHECK(rows.offsets == std::vector<std::size_t>{0, 2, 4, 5});
}

TEST_CASE("lag shifts within each group", "[baselines]") {
  const std::vector<double> v{1, 10, 2, 20, 3};
  const std::vector<std::int64_t> g{0, 1, 0, 1, 0};
  const auto out = paudit::lag(v, std::span<const std::int64_t>(g), 1);
  const std::vector<double> expected{kNaN, kNaN, 1, 10, 2};
  for (std::size_t i = 0; i < v.size(); ++i) CHECK(same(out[i], expected[i]));
  CHECK_THROWS_AS(paudit::lag(v, std::nullopt, -1), std::invalid_argument);
}

TEST_CASE("rolling mean needs a full window and skips NaN", "[baselines]") {
  const std::vector<double> v{1, 2, 3, kNaN, 5, 6};
  const auto out = paudit::rolling_mean(v, std::nullopt, 3, 3);
  CHECK(std::isnan(out[0]));
  CHECK(std::isnan(out[1]));
  CHECK_THAT(out[2], WithinAbs(2.0, 1e-15));
  CHECK(std::isnan(out[3]));  // only two observations in the window
  const auto loose = paudit::rolling_mean(v, std::nullopt, 3, 2);
  CHECK_THAT(loose[3], WithinAbs(2.5, 1e-15));
  CHECK_THAT(loose[5], WithinAbs(5.5, 1e-15));
  // A constant window returns the constant exactly (pandas GH#42064 guard).
  const std::vector<double> c(10, 0.1);
  const auto flat = paudit::rolling_mean(c, std::nullopt, 4, 4);
  CHECK(flat[9] == 0.1);
}

TEST_CASE("ewm with adjust=False matches the recursive definition", "[baselines]") {
  const std::vector<double> v{1.0, 2.0, 4.0};
  const auto out = paudit::ewm_mean(v, std::nullopt, 4.0);  // alpha = 2 / (span + 1) = 0.4
  CHECK(out[0] == 1.0);
  CHECK_THAT(out[1], WithinAbs(0.6 * 1.0 + 0.4 * 2.0, 1e-15));
  CHECK_THAT(out[2], WithinAbs(0.6 * out[1] + 0.4 * 4.0, 1e-15));
}

TEST_CASE("linear AR recovers an exact AR(1)", "[baselines]") {
  std::vector<double> y(200);
  y[0] = 1.0;
  for (std::size_t i = 1; i < y.size(); ++i) y[i] = 0.8 * y[i - 1] + std::sin(static_cast<double>(i));
  const std::vector<std::uint8_t> fit(y.size(), 1);
  const auto pred = paudit::linear_ar(y, std::nullopt, 1, fit, 2, 1e-9, false);
  CHECK(std::isnan(pred[0]));
  CHECK(std::isnan(pred[1]));
  CHECK(std::isfinite(pred[2]));
}

TEST_CASE("season estimate finds a clean period", "[baselines]") {
  std::vector<double> y(1000);
  for (std::size_t i = 0; i < y.size(); ++i)
    y[i] = std::sin(2.0 * std::numbers::pi * static_cast<double>(i) / 50.0);
  const std::vector<std::uint8_t> fit(y.size(), 1);
  const auto season = paudit::estimate_season(y, std::nullopt, fit);
  REQUIRE(season.has_value());
  CHECK(*season == 50);
  std::vector<double> walk(1000);
  for (std::size_t i = 1; i < walk.size(); ++i) walk[i] = walk[i - 1] + 1.0;
  CHECK_FALSE(paudit::estimate_season(walk, std::nullopt, fit).has_value());
}
