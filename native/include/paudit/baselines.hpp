#pragma once

// Causal baselines built from the target column alone (predictive_audit.baselines).
// Row t holds target[t] = y[t + h]; rows are in time order within each group. Grouped
// operations follow pandas `groupby(groups, sort=False)`: groups in order of first appearance,
// rows within a group in their original order, results written back to the original rows.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "paudit/bootstrap.hpp"

namespace paudit {

/// Rows of each group in pandas `groupby(sort=False)` order, as compressed rows (CSR).
struct GroupRows {
  std::vector<std::size_t> rows;     ///< row indices, group by group
  std::vector<std::size_t> offsets;  ///< group g owns rows[offsets[g] .. offsets[g + 1])
  [[nodiscard]] std::size_t n_groups() const noexcept { return offsets.empty() ? 0 : offsets.size() - 1; }
};

/// One group of all rows when `groups` is empty.
[[nodiscard]] GroupRows group_rows(std::size_t n, GroupCodes groups);

/// `values.groupby(groups).shift(k)` for k >= 0 (NaN where a group has no history yet).
[[nodiscard]] std::vector<double> lag(std::span<const double> values, GroupCodes groups, std::int64_t k);

/// `values.groupby(groups).rolling(window, min_periods).mean()`, bit-identical to pandas
/// (Kahan add/remove sums, constant-window and sign guards; +-inf treated as NaN).
[[nodiscard]] std::vector<double> rolling_mean(std::span<const double> values, GroupCodes groups,
                                               std::size_t window, std::size_t min_periods);

/// `values.groupby(groups).ewm(span=span, adjust=False, min_periods=1).mean()` (ignore_na=False).
[[nodiscard]] std::vector<double> ewm_mean(std::span<const double> values, GroupCodes groups, double span);

/// predictive_audit.baselines.linear_ar: causal Ridge AR(p) on observed lags [h, h + p), with an
/// unpenalized intercept, fit on `fit_mask` rows; per-group weights when a group has at least
/// p + 5 fit rows, pooled weights otherwise. Throws std::runtime_error if a system is singular.
[[nodiscard]] std::vector<double> linear_ar(std::span<const double> target, GroupCodes groups,
                                            std::int64_t horizon, std::span<const std::uint8_t> fit_mask,
                                            std::size_t lags, double alpha, bool clip_unit_interval);

/// predictive_audit.baselines.estimate_season: dominant period from the pooled FFT
/// autocorrelation of fit rows, or nullopt when there is no clear seasonal peak.
[[nodiscard]] std::optional<std::int64_t> estimate_season(std::span<const double> target, GroupCodes groups,
                                                          std::span<const std::uint8_t> fit_mask,
                                                          std::int64_t max_lag = 2000,
                                                          std::int64_t min_lag = 2);

}  // namespace paudit
