#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "paudit/metrics.hpp"

namespace paudit {

/// Integer series codes, one per row, contiguous within each series (rows sorted by group).
using GroupCodes = std::optional<std::span<const std::int64_t>>;
/// Per-row time keys: none, integers (incl. datetime64 viewed as int64) or floats.
using TimeKeys = std::variant<std::monostate, std::span<const std::int64_t>, std::span<const double>>;

/// Result of predictive_audit.metrics._is_aligned_panel.
struct PanelInfo {
  bool aligned = false;
  std::vector<std::int64_t> time_index;  ///< dense code per row in [0, n_times) when aligned
  std::size_t n_times = 0;
};

[[nodiscard]] PanelInfo detect_aligned_panel(GroupCodes groups, const TimeKeys& times);

/// Result of predictive_audit.metrics._group_circular_layout.
struct CircularLayout {
  std::vector<std::int64_t> gather;  ///< row index for each padded position
  std::vector<std::int64_t> starts;  ///< valid block starts (one per original row)
};

[[nodiscard]] CircularLayout group_circular_layout(std::span<const std::int64_t> groups, std::size_t pad);

/// predictive_audit.metrics.default_block_length.
[[nodiscard]] std::size_t default_block_length(std::size_t n_total, GroupCodes groups, const TimeKeys& times);

/// Politis-White (2004) / Patton-Politis-White (2009) optimal circular-block-bootstrap length.
[[nodiscard]] std::size_t estimate_block_length(std::span<const double> x,
                                                std::optional<std::size_t> max_block = std::nullopt);

struct Interval {
  double lo;
  double hi;
};

/// 95% circular-block-bootstrap interval of score(model) - score(baseline), identical to
/// predictive_audit.metrics.block_bootstrap_diff (same seed, same draws). Rank metrics
/// (roc_auc, pr_auc) require 0/1 labels and finite scores.
/// Bootstrap draws are evaluated in parallel; results do not depend on the thread count.
[[nodiscard]] Interval block_bootstrap_diff(Metric metric, std::span<const double> y_true,
                                            std::span<const double> model, std::span<const double> baseline,
                                            std::int64_t block, std::int64_t n_boot, std::uint64_t seed,
                                            GroupCodes groups = std::nullopt, const TimeKeys& times = {},
                                            double threshold = 0.5);

}  // namespace paudit
