#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "paudit/rank.hpp"

using Catch::Matchers::WithinAbs;
using paudit::Metric;

TEST_CASE("ROC-AUC counts concordant pairs, ties count half", "[rank]") {
  const std::vector<double> y{1, 0, 1, 0};
  CHECK_THAT(paudit::roc_auc(y, std::vector<double>{0.9, 0.1, 0.8, 0.2}), WithinAbs(1.0, 1e-15));
  CHECK_THAT(paudit::roc_auc(y, std::vector<double>{0.5, 0.5, 0.5, 0.5}), WithinAbs(0.5, 1e-15));
  CHECK(std::isnan(paudit::roc_auc(std::vector<double>{1, 1}, std::vector<double>{0.1, 0.2})));
}

TEST_CASE("average precision on a hand-worked example", "[rank]") {
  // sklearn docs: y=[0,0,1,1], scores=[0.1,0.4,0.35,0.8] -> 0.8333...
  const std::vector<double> y{0, 0, 1, 1};
  const std::vector<double> s{0.1, 0.4, 0.35, 0.8};
  CHECK_THAT(paudit::average_precision(y, s), WithinAbs(0.8333333333333333, 1e-15));
}

TEST_CASE("weights equal explicit repetition", "[rank]") {
  const std::vector<double> y{1, 0, 1, 0, 1};
  const std::vector<double> s{0.9, 0.7, 0.4, 0.4, 0.2};
  const std::vector<std::uint32_t> w{2, 0, 1, 3, 1};
  const std::vector<double> y_rep{1, 1, 1, 0, 0, 0, 1};
  const std::vector<double> s_rep{0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.2};
  const auto order = paudit::score_order(s);
  CHECK(paudit::roc_auc(y, order, w) == paudit::roc_auc(y_rep, s_rep));
  CHECK(paudit::average_precision(y, order, w) == paudit::average_precision(y_rep, s_rep));
}

TEST_CASE("optimal threshold lifts a skewed F1", "[rank]") {
  std::vector<double> y(200, 0.0);
  std::vector<double> s(200);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y[i] = i % 10 == 0 ? 1.0 : 0.0;
    s[i] = y[i] == 1.0 ? 0.3 + 0.001 * static_cast<double>(i) : 0.001 * static_cast<double>(i % 97);
  }
  const double t = paudit::optimal_threshold(Metric::f1, y, s);
  CHECK(t > 0.09);
  CHECK(t <= 0.3);
  CHECK(paudit::optimal_threshold(Metric::mae, y, s) == 0.5);
}
