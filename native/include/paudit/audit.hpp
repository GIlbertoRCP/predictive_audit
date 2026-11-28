#pragma once

// The full audit pipeline (predictive_audit.audit.audit / batch_audit) in C++: sort, group,
// baselines, metric table, and every built-in check, producing the same report, including the
// same human-readable summaries, as the Python reference.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "paudit/metrics.hpp"

namespace paudit {

enum class Task : std::uint8_t { classification, regression };
/// Check severities, ordered as predictive_audit.audit.SEVERITY.
enum class Status : std::uint8_t { info = 0, pass = 1, warn = 2, fail = 3 };

[[nodiscard]] std::string_view status_name(Status s) noexcept;

/// One value in a check's `details` mapping (the shapes the Python report uses).
using NamedDoubles = std::vector<std::pair<std::string, double>>;
using DetailValue = std::variant<std::monostate, double, std::int64_t, std::string, std::vector<double>,
                                 std::vector<std::string>, NamedDoubles>;
using Details = std::vector<std::pair<std::string, DetailValue>>;

struct Check {
  std::string name;
  Status status = Status::info;
  std::string summary;
  Details details;
};

struct AuditReport {
  Task task = Task::regression;
  std::string target;
  std::int64_t horizon = 1;
  Metric metric = Metric::mae;
  std::optional<std::string> model;
  std::optional<std::int64_t> season;
  std::size_t n_eval = 0;
  std::optional<double> prevalence;
  std::string best_baseline;
  /// (prediction name, [(metric, value)]) in report order.
  std::vector<std::pair<std::string, std::vector<std::pair<Metric, double>>>> table;
  std::vector<Check> checks;
  std::optional<double> threshold;
  std::optional<double> mase;

  /// Original row index of every scored row, in sorted order, and the scored values of every
  /// prediction (same order as `table`), for Python-side extra checks. Filled on request.
  std::vector<std::int64_t> eval_rows;
  std::vector<std::vector<double>> eval_predictions;
  std::vector<double> eval_target;

  [[nodiscard]] Status verdict() const noexcept;
};

/// A sort key column: integer values/codes or floats (no NaN), compared by value.
using KeyColumn = std::variant<std::span<const std::int64_t>, std::span<const double>>;

/// Per-row split label: 0 = other (history only), 1 = train, 2 = test.
enum SplitKind : std::uint8_t { kSplitOther = 0, kSplitTrain = 1, kSplitTest = 2 };

struct NamedColumn {
  std::string name;
  std::span<const double> values;
};

/// Columns in the caller's original row order.
struct AuditColumns {
  std::span<const double> target;
  std::optional<std::span<const double>> pred;
  std::vector<KeyColumn> group_keys;  ///< sorted first, in order
  std::optional<KeyColumn> time;      ///< sorted after the group keys
  std::optional<std::span<const std::uint8_t>> split;
  std::vector<NamedColumn> features;
  std::vector<NamedColumn> extra_baselines;
};

struct AuditConfig {
  Task task = Task::regression;
  std::int64_t horizon = 1;
  std::string target_name;
  std::optional<std::string> model_name;
  std::string split_name;  ///< for error messages
  Metric metric = Metric::mae;
  enum class SeasonMode : std::uint8_t { none, fixed, automatic } season_mode = SeasonMode::none;
  std::int64_t season = 0;
  /// Python's message when `season` had an unsupported type, raised at the same point it would be.
  std::optional<std::string> season_error;
  std::optional<std::int64_t> window;
  bool include_ewma = false;
  std::optional<std::int64_t> ar_lags;
  std::optional<double> reported;
  bool threshold_auto = false;
  double threshold = 0.5;
  std::int64_t n_boot = 500;
  enum class BlockMode : std::uint8_t {
    automatic_default,
    fixed,
    politis_white
  } block_mode = BlockMode::automatic_default;
  std::int64_t block = 0;
  std::uint64_t seed = 0;
  bool ewm_com1_update = true;  ///< pandas >= 3.0 EWM weighting (see ewm_mean)
  bool want_eval_arrays = false;
};

/// predictive_audit.audit.audit. Throws std::invalid_argument with the Python message for
/// data-dependent errors (non-binary classification target, no test rows, too few scored rows).
[[nodiscard]] AuditReport audit(const AuditColumns& columns, const AuditConfig& config);

/// One audit per series, run in parallel. `series` gives each row's series index (>= 0) or -1 to
/// skip the row; series run in index order. Throws the first series' error, as Python would.
[[nodiscard]] std::vector<AuditReport> audit_each_series(const AuditColumns& columns,
                                                         const AuditConfig& config,
                                                         std::span<const std::int64_t> series,
                                                         std::size_t n_series);

}  // namespace paudit
