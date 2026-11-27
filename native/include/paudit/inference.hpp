#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "paudit/bootstrap.hpp"
#include "paudit/metrics.hpp"

namespace paudit {

struct DmResult {
  double stat;
  double p_value;
};

/// Diebold-Mariano test with Newey-West (Bartlett) HAC variance and the Harvey-Leybourne-Newbold
/// finite-sample correction. Positive statistic means `model` beats `baseline`. Returns NaNs for
/// metrics without a per-step loss (r2, f1, mcc, balanced_accuracy, roc_auc, pr_auc) or < 5 steps.
[[nodiscard]] DmResult diebold_mariano(Metric metric, std::span<const double> y_true,
                                       std::span<const double> model, std::span<const double> baseline,
                                       std::int64_t horizon = 1, GroupCodes groups = std::nullopt,
                                       const TimeKeys& times = {}, double threshold = 0.5);

/// Benjamini-Hochberg FDR q-values; NaN inputs stay NaN.
[[nodiscard]] std::vector<double> benjamini_hochberg(std::span<const double> p_values);
/// Holm-Bonferroni step-down adjusted p-values; NaN inputs stay NaN.
[[nodiscard]] std::vector<double> holm_bonferroni(std::span<const double> p_values);

/// Two-sided Student-t p-value, P(|T| >= |t|) with `df` degrees of freedom.
[[nodiscard]] double student_t_two_sided_p(double t, double df);
/// Regularized incomplete beta I_x(a, b); `y` must equal 1 - x (passed separately for accuracy).
[[nodiscard]] double incomplete_beta(double a, double b, double x, double y);

}  // namespace paudit
