#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "paudit/metrics.hpp"

using Catch::Matchers::WithinAbs;
using paudit::Metric;

TEST_CASE("metric names round-trip", "[metrics]") {
  for (const char* name : {"f1", "pr_auc", "mcc", "balanced_accuracy", "accuracy", "roc_auc", "brier", "mae",
                           "rmse", "r2", "smape"}) {
    const auto m = paudit::parse_metric(name);
    REQUIRE(m.has_value());
    CHECK(paudit::metric_name(*m) == name);
  }
  CHECK_FALSE(paudit::parse_metric("nonexistent").has_value());
}

TEST_CASE("classification metrics on a hand-checked confusion matrix", "[metrics]") {
  // tp=2, fp=1, fn=1, tn=2
  const std::vector<double> y{1, 1, 1, 0, 0, 0};
  const std::vector<double> p{0.9, 0.8, 0.1, 0.7, 0.2, 0.3};
  CHECK_THAT(paudit::compute(Metric::f1, y, p), WithinAbs(2.0 / 3.0, 1e-15));
  CHECK_THAT(paudit::compute(Metric::accuracy, y, p), WithinAbs(4.0 / 6.0, 1e-15));
  CHECK_THAT(paudit::compute(Metric::balanced_accuracy, y, p), WithinAbs(2.0 / 3.0, 1e-15));
  CHECK_THAT(paudit::compute(Metric::mcc, y, p), WithinAbs(1.0 / 3.0, 1e-15));
  CHECK_THAT(paudit::compute(Metric::roc_auc, y, p), WithinAbs(6.0 / 9.0, 1e-15));
  CHECK_THROWS_AS(paudit::compute(Metric::roc_auc, std::vector<double>{0, 2}, std::vector<double>{0.1, 0.2}),
                  std::invalid_argument);
}

TEST_CASE("regression metrics", "[metrics]") {
  const std::vector<double> y{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> p{1.5, 2.0, 2.0, 5.0};
  CHECK_THAT(paudit::compute(Metric::mae, y, p), WithinAbs(0.625, 1e-15));
  CHECK_THAT(paudit::compute(Metric::rmse, y, p), WithinAbs(std::sqrt(2.25 / 4.0), 1e-15));
  CHECK_THAT(paudit::compute(Metric::r2, y, p), WithinAbs(1.0 - 2.25 / 5.0, 1e-15));
  CHECK(paudit::compute(Metric::smape, std::vector<double>{0, 0}, std::vector<double>{0, 0}) == 0.0);
  CHECK(paudit::compute(Metric::r2, std::vector<double>{2, 2}, std::vector<double>{2, 2}) == 1.0);
  CHECK(paudit::score(Metric::mae, y, p) == -paudit::compute(Metric::mae, y, p));
  CHECK_THROWS_AS(paudit::compute(Metric::mae, y, std::vector<double>{1.0}), std::invalid_argument);
}
