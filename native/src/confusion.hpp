#pragma once

// Confusion-matrix metrics from (tp, fp, fn, tn) counts, written exactly as
// predictive_audit.metrics evaluates them (same operations, same order).

#include <cmath>

namespace paudit::detail {

[[nodiscard]] inline double f1_from(double tp, double fp, double fn) noexcept {
  const double d = 2.0 * tp + fp + fn;
  return d > 0.0 ? 2.0 * tp / d : 0.0;
}

[[nodiscard]] inline double balanced_accuracy_from(double tp, double fp, double fn, double tn) noexcept {
  const double pos = tp + fn;
  const double neg = tn + fp;
  const double tpr = pos > 0.0 ? tp / pos : 0.0;
  const double tnr = neg > 0.0 ? tn / neg : 0.0;
  if (pos > 0.0 && neg > 0.0) return 0.5 * (tpr + tnr);
  if (pos > 0.0 && neg == 0.0) return tpr;
  return tnr;
}

[[nodiscard]] inline double mcc_from(double tp, double fp, double fn, double tn) noexcept {
  const double d = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return d > 0.0 ? (tp * tn - fp * fn) / d : 0.0;
}

}  // namespace paudit::detail
