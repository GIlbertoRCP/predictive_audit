#pragma once

// Rank metrics (ROC-AUC, average precision) that reproduce scikit-learn bit for bit, plus the
// sort-once machinery that turns their bootstrap into an O(n) weighted sweep per draw.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "paudit/metrics.hpp"

namespace paudit {

/// Rows sorted by descending score (stable), cut into runs of tied scores.
struct ScoreOrder {
  std::vector<std::uint32_t> order;     ///< row indices, highest score first
  std::vector<std::uint32_t> tie_ends;  ///< exclusive end (in `order`) of each run of equal scores
};

[[nodiscard]] ScoreOrder score_order(std::span<const double> scores);

/// sklearn.metrics.roc_auc_score / average_precision_score for 0/1 labels and finite scores.
/// `weights` (optional) are integer multiplicities: the result equals the sklearn score on the
/// array where row i is repeated weights[i] times. NaN when only one class is present.
[[nodiscard]] double roc_auc(std::span<const double> y_true, const ScoreOrder& order,
                             std::span<const std::uint32_t> weights = {});
[[nodiscard]] double average_precision(std::span<const double> y_true, const ScoreOrder& order,
                                       std::span<const std::uint32_t> weights = {});

/// Throws std::invalid_argument unless labels are exactly 0/1 and every score is finite
/// (scikit-learn raises on non-finite scores; other labels change its positive class).
void validate_rank_inputs(std::span<const double> y_true, std::span<const double> scores);

/// Convenience overloads that validate inputs and sort `scores` themselves.
[[nodiscard]] double roc_auc(std::span<const double> y_true, std::span<const double> scores);
[[nodiscard]] double average_precision(std::span<const double> y_true, std::span<const double> scores);

/// predictive_audit.metrics.optimal_threshold: the classification threshold that maximizes a
/// thresholded metric over unique scores (or 97 quantiles of them), in O(n log n) total.
[[nodiscard]] double optimal_threshold(Metric metric, std::span<const double> y_true,
                                       std::span<const double> y_score);

}  // namespace paudit
