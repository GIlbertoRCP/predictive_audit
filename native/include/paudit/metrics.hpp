#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace paudit {

/// Every metric the audit understands (same names as predictive_audit.metrics).
enum class Metric : std::uint8_t {
  f1,
  pr_auc,
  mcc,
  balanced_accuracy,
  accuracy,
  roc_auc,
  brier,
  mae,
  rmse,
  r2,
  smape,
};

[[nodiscard]] std::optional<Metric> parse_metric(std::string_view name) noexcept;
[[nodiscard]] std::string_view metric_name(Metric m) noexcept;

/// Error metrics (mae, rmse, smape, brier): lower raw values are better.
[[nodiscard]] bool lower_is_better(Metric m) noexcept;
/// Metrics whose bootstrap uses O(1)-per-block prefix sums of per-row sufficient statistics.
[[nodiscard]] bool is_prefix_metric(Metric m) noexcept;
/// Metrics computed natively by `compute` (all of them; rank metrics need 0/1 labels).
[[nodiscard]] bool has_native_compute(Metric m) noexcept;

/// Raw metric value, matching predictive_audit.metrics.compute. Classification targets are
/// thresholded at 0.5, predictions at `threshold`. Rank metrics (roc_auc, pr_auc) match
/// scikit-learn and return NaN with one class; they throw std::invalid_argument for labels other
/// than 0/1 or non-finite scores. Throws for inputs of different lengths.
[[nodiscard]] double compute(Metric m, std::span<const double> y_true, std::span<const double> y_pred,
                             double threshold = 0.5);

/// Higher-is-better orientation of `compute` (error metrics negated).
[[nodiscard]] double score(Metric m, std::span<const double> y_true, std::span<const double> y_pred,
                           double threshold = 0.5);

}  // namespace paudit
