// nanobind extension: predictive_audit._native
//
// Thin, zero-copy bindings over libpaudit. Inputs arrive as contiguous float64 / int64 NumPy
// arrays (the Python wrappers in predictive_audit.metrics normalize them), and the GIL is
// released for every kernel so other Python threads keep running.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "paudit/audit.hpp"
#include "paudit/baselines.hpp"
#include "paudit/bootstrap.hpp"
#include "paudit/inference.hpp"
#include "paudit/metrics.hpp"
#include "paudit/parallel.hpp"
#include "paudit/rank.hpp"
#include "paudit/rng.hpp"
#include "paudit/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using F64 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64 = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U8 = nb::ndarray<const std::uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using NumpyOut = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

std::span<const double> span_of(const F64& a) { return {a.data(), a.shape(0)}; }
std::span<const std::int64_t> span_of(const I64& a) { return {a.data(), a.shape(0)}; }
std::span<const std::uint8_t> span_of(const U8& a) { return {a.data(), a.shape(0)}; }

paudit::Metric metric_of(const std::string& name) {
  const auto m = paudit::parse_metric(name);
  if (!m) throw std::invalid_argument("unknown metric '" + name + "'");
  return *m;
}

paudit::GroupCodes groups_of(const std::optional<I64>& g) {
  if (!g) return std::nullopt;
  return span_of(*g);
}

paudit::TimeKeys times_of(const std::optional<I64>& ti, const std::optional<F64>& tf) {
  if (ti && tf) throw std::invalid_argument("pass at most one of times_i64 / times_f64");
  if (ti) return span_of(*ti);
  if (tf) return span_of(*tf);
  return std::monostate{};
}

template <class T>
NumpyOut<T> to_numpy(std::vector<T>&& v) {
  auto* heap = new std::vector<T>(std::move(v));
  const nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return NumpyOut<T>(heap->data(), {heap->size()}, owner);
}

// --- audit marshalling --------------------------------------------------------------------------

/// Python-owned arrays backing an AuditColumns (kept alive for the duration of the call).
struct AuditInputs {
  F64 target;
  std::optional<F64> pred;
  nb::list group_keys;  // each an int64 or float64 1-D array
  std::optional<I64> time_i64;
  std::optional<F64> time_f64;
  std::optional<U8> split;
  std::vector<std::string> feature_names;
  std::vector<F64> features;
  std::vector<std::string> extra_names;
  std::vector<F64> extras;
};

paudit::KeyColumn key_column_of(nb::handle h) {
  I64 ints;
  if (nb::try_cast(h, ints, false)) return span_of(ints);
  F64 floats;
  if (nb::try_cast(h, floats, false)) return span_of(floats);
  throw std::invalid_argument("group keys must be contiguous int64 or float64 arrays");
}

paudit::AuditColumns columns_of(const AuditInputs& in) {
  paudit::AuditColumns c;
  c.target = span_of(in.target);
  if (in.pred) c.pred = span_of(*in.pred);
  for (nb::handle h : in.group_keys) c.group_keys.push_back(key_column_of(h));
  if (in.time_i64 && in.time_f64) throw std::invalid_argument("pass at most one of time_i64 / time_f64");
  if (in.time_i64) c.time = paudit::KeyColumn(span_of(*in.time_i64));
  if (in.time_f64) c.time = paudit::KeyColumn(span_of(*in.time_f64));
  if (in.split) c.split = span_of(*in.split);
  if (in.feature_names.size() != in.features.size() || in.extra_names.size() != in.extras.size()) {
    throw std::invalid_argument("names and columns must have the same length");
  }
  for (std::size_t k = 0; k < in.features.size(); ++k)
    c.features.push_back({in.feature_names[k], span_of(in.features[k])});
  for (std::size_t k = 0; k < in.extras.size(); ++k)
    c.extra_baselines.push_back({in.extra_names[k], span_of(in.extras[k])});
  return c;
}

nb::object detail_to_py(const paudit::DetailValue& v) {
  return std::visit(
      [](const auto& x) -> nb::object {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<std::string>>) {
          nb::list out;
          for (const auto& e : x) out.append(nb::cast(e));
          return out;
        } else if constexpr (std::is_same_v<T, paudit::NamedDoubles>) {
          nb::dict out;
          for (const auto& [k, e] : x) out[nb::str(k.c_str())] = e;
          return out;
        } else {
          return nb::cast(x);
        }
      },
      v);
}

nb::object optional_to_py(const auto& opt) { return opt ? nb::cast(*opt) : nb::none(); }

nb::dict report_to_py(paudit::AuditReport&& r) {
  nb::dict d;
  d["task"] = r.task == paudit::Task::classification ? "classification" : "regression";
  d["target"] = r.target;
  d["horizon"] = r.horizon;
  d["metric"] = std::string(paudit::metric_name(r.metric));
  d["model"] = optional_to_py(r.model);
  d["season"] = optional_to_py(r.season);
  d["n_eval"] = r.n_eval;
  d["prevalence"] = optional_to_py(r.prevalence);
  d["best_baseline"] = r.best_baseline;
  nb::dict table;
  for (const auto& [name, row] : r.table) {
    nb::dict values;
    for (const auto& [metric, v] : row) values[nb::str(std::string(paudit::metric_name(metric)).c_str())] = v;
    table[nb::str(name.c_str())] = values;
  }
  d["table"] = table;
  nb::list checks;
  for (const auto& c : r.checks) {
    nb::dict details;
    for (const auto& [k, v] : c.details) details[nb::str(k.c_str())] = detail_to_py(v);
    checks.append(nb::make_tuple(c.name, std::string(paudit::status_name(c.status)), c.summary, details));
  }
  d["checks"] = checks;
  d["threshold"] = optional_to_py(r.threshold);
  d["mase"] = optional_to_py(r.mase);
  if (!r.eval_rows.empty()) {
    d["eval_rows"] = to_numpy(std::move(r.eval_rows));
    d["eval_target"] = to_numpy(std::move(r.eval_target));
    nb::list preds;
    for (auto& p : r.eval_predictions) preds.append(to_numpy(std::move(p)));
    d["eval_predictions"] = preds;
  }
  return d;
}

paudit::AuditConfig config_of(const nb::dict& cfg) {
  paudit::AuditConfig c;
  const std::string task = nb::cast<std::string>(cfg["task"]);
  c.task = task == "classification" ? paudit::Task::classification : paudit::Task::regression;
  c.horizon = nb::cast<std::int64_t>(cfg["horizon"]);
  c.target_name = nb::cast<std::string>(cfg["target_name"]);
  if (!cfg["model_name"].is_none()) c.model_name = nb::cast<std::string>(cfg["model_name"]);
  c.split_name = nb::cast<std::string>(cfg["split_name"]);
  c.metric = metric_of(nb::cast<std::string>(cfg["metric"]));
  const std::string season_mode = nb::cast<std::string>(cfg["season_mode"]);
  c.season_mode = season_mode == "auto"    ? paudit::AuditConfig::SeasonMode::automatic
                  : season_mode == "fixed" ? paudit::AuditConfig::SeasonMode::fixed
                                           : paudit::AuditConfig::SeasonMode::none;
  c.season = nb::cast<std::int64_t>(cfg["season"]);
  if (!cfg["season_error"].is_none()) c.season_error = nb::cast<std::string>(cfg["season_error"]);
  if (!cfg["window"].is_none()) c.window = nb::cast<std::int64_t>(cfg["window"]);
  c.include_ewma = nb::cast<bool>(cfg["include_ewma"]);
  if (!cfg["ar_lags"].is_none()) c.ar_lags = nb::cast<std::int64_t>(cfg["ar_lags"]);
  if (!cfg["reported"].is_none()) c.reported = nb::cast<double>(cfg["reported"]);
  c.threshold_auto = nb::cast<bool>(cfg["threshold_auto"]);
  c.threshold = nb::cast<double>(cfg["threshold"]);
  c.n_boot = nb::cast<std::int64_t>(cfg["n_boot"]);
  const std::string block_mode = nb::cast<std::string>(cfg["block_mode"]);
  c.block_mode = block_mode == "auto"    ? paudit::AuditConfig::BlockMode::politis_white
                 : block_mode == "fixed" ? paudit::AuditConfig::BlockMode::fixed
                                         : paudit::AuditConfig::BlockMode::automatic_default;
  c.block = nb::cast<std::int64_t>(cfg["block"]);
  c.seed = nb::cast<std::uint64_t>(cfg["seed"]);
  c.ewm_com1_update = nb::cast<bool>(cfg["ewm_com1_update"]);
  c.want_eval_arrays = nb::cast<bool>(cfg["want_eval_arrays"]);
  return c;
}

}  // namespace

NB_MODULE(_native, m) {
  m.doc() = "Native C++ engine for predictive-audit.";

  // --- build metadata & threading ----------------------------------------------------------
  m.def(
      "version", [] { return std::string{paudit::version()}; },
      "Version of the native engine (matches the Python package version).");

  m.def(
      "build_info",
      [] {
        const auto info = paudit::build_info();
        nb::dict d;
        d["version"] = std::string{info.version};
        d["compiler"] = std::string{info.compiler};
        d["build_type"] = std::string{info.build_type};
        d["cplusplus"] = info.cplusplus;
        d["hardware_threads"] = info.hardware_threads;
        return d;
      },
      "Compiler, C++ standard, build type and hardware thread count of this build.");

  m.def("num_threads", &paudit::parallel::num_threads, "Threads used by native parallel kernels.");
  m.def("set_num_threads", &paudit::parallel::set_num_threads, "n"_a,
        "Set the native thread count (0 restores the default: PREDICTIVE_AUDIT_THREADS or all cores).");

  // --- metrics ---------------------------------------------------------------------------------
  m.def(
      "compute",
      [](const std::string& metric, const F64& y_true, const F64& y_pred, double threshold) {
        const auto mt = metric_of(metric);
        const nb::gil_scoped_release release;
        return paudit::compute(mt, span_of(y_true), span_of(y_pred), threshold);
      },
      "metric"_a, "y_true"_a, "y_pred"_a, "threshold"_a = 0.5);

  m.def("has_native_compute", [](const std::string& metric) {
    const auto mt = paudit::parse_metric(metric);
    return mt.has_value() && paudit::has_native_compute(*mt);
  });

  // --- bootstrap ---------------------------------------------------------------------------------
  m.def(
      "block_bootstrap_diff",
      [](const std::string& metric, const F64& y_true, const F64& model, const F64& baseline,
         std::int64_t block, std::int64_t n_boot, std::uint64_t seed, const std::optional<I64>& groups,
         const std::optional<I64>& times_i64, const std::optional<F64>& times_f64, double threshold) {
        const auto mt = metric_of(metric);
        const auto times = times_of(times_i64, times_f64);
        const nb::gil_scoped_release release;
        const auto r = paudit::block_bootstrap_diff(mt, span_of(y_true), span_of(model), span_of(baseline),
                                                    block, n_boot, seed, groups_of(groups), times, threshold);
        return std::make_pair(r.lo, r.hi);
      },
      "metric"_a, "y_true"_a, "model"_a, "baseline"_a, "block"_a, "n_boot"_a, "seed"_a,
      "groups"_a = nb::none(), "times_i64"_a = nb::none(), "times_f64"_a = nb::none(), "threshold"_a = 0.5);

  m.def(
      "estimate_block_length",
      [](const F64& x, std::optional<std::size_t> max_block) {
        const nb::gil_scoped_release release;
        return paudit::estimate_block_length(span_of(x), max_block);
      },
      "x"_a, "max_block"_a = nb::none());

  m.def(
      "default_block_length",
      [](std::size_t n_total, const std::optional<I64>& groups, const std::optional<I64>& times_i64,
         const std::optional<F64>& times_f64) {
        const auto times = times_of(times_i64, times_f64);
        const nb::gil_scoped_release release;
        return paudit::default_block_length(n_total, groups_of(groups), times);
      },
      "n_total"_a, "groups"_a = nb::none(), "times_i64"_a = nb::none(), "times_f64"_a = nb::none());

  m.def(
      "is_aligned_panel",
      [](const std::optional<I64>& groups, const std::optional<I64>& times_i64,
         const std::optional<F64>& times_f64) {
        const auto times = times_of(times_i64, times_f64);
        paudit::PanelInfo info;
        {
          const nb::gil_scoped_release release;
          info = paudit::detect_aligned_panel(groups_of(groups), times);
        }
        nb::object inv = nb::none();
        if (info.aligned) inv = nb::cast(to_numpy(std::move(info.time_index)));
        return std::make_tuple(info.aligned, inv, info.n_times);
      },
      "groups"_a.none(), "times_i64"_a = nb::none(), "times_f64"_a = nb::none());

  m.def(
      "group_circular_layout",
      [](const I64& groups, std::size_t pad) {
        auto layout = paudit::group_circular_layout(span_of(groups), pad);
        return std::make_pair(to_numpy(std::move(layout.gather)), to_numpy(std::move(layout.starts)));
      },
      "groups"_a, "pad"_a);

  // --- inference ----------------------------------------------------------------------------------
  m.def(
      "diebold_mariano",
      [](const std::string& metric, const F64& y_true, const F64& model, const F64& baseline,
         std::int64_t horizon, const std::optional<I64>& groups, const std::optional<I64>& times_i64,
         const std::optional<F64>& times_f64, double threshold) {
        const auto mt = metric_of(metric);
        const auto times = times_of(times_i64, times_f64);
        const nb::gil_scoped_release release;
        const auto r = paudit::diebold_mariano(mt, span_of(y_true), span_of(model), span_of(baseline),
                                               horizon, groups_of(groups), times, threshold);
        return std::make_pair(r.stat, r.p_value);
      },
      "metric"_a, "y_true"_a, "model"_a, "baseline"_a, "horizon"_a = 1, "groups"_a = nb::none(),
      "times_i64"_a = nb::none(), "times_f64"_a = nb::none(), "threshold"_a = 0.5);

  m.def(
      "benjamini_hochberg", [](const F64& p) { return to_numpy(paudit::benjamini_hochberg(span_of(p))); },
      "p"_a);
  m.def("holm_bonferroni", [](const F64& p) { return to_numpy(paudit::holm_bonferroni(span_of(p))); }, "p"_a);
  m.def("student_t_two_sided_p", &paudit::student_t_two_sided_p, "t"_a, "df"_a);

  // --- baselines ----------------------------------------------------------------------------------
  m.def(
      "lag",
      [](const F64& values, const std::optional<I64>& groups, std::int64_t k) {
        std::vector<double> out;
        {
          const nb::gil_scoped_release release;
          out = paudit::lag(span_of(values), groups_of(groups), k);
        }
        return to_numpy(std::move(out));
      },
      "values"_a, "groups"_a.none(), "k"_a);

  m.def(
      "rolling_mean",
      [](const F64& values, const std::optional<I64>& groups, std::size_t window, std::size_t min_periods) {
        std::vector<double> out;
        {
          const nb::gil_scoped_release release;
          out = paudit::rolling_mean(span_of(values), groups_of(groups), window, min_periods);
        }
        return to_numpy(std::move(out));
      },
      "values"_a, "groups"_a.none(), "window"_a, "min_periods"_a);

  m.def(
      "ewm_mean",
      [](const F64& values, const std::optional<I64>& groups, double span, bool com1_update) {
        std::vector<double> out;
        {
          const nb::gil_scoped_release release;
          out = paudit::ewm_mean(span_of(values), groups_of(groups), span, com1_update);
        }
        return to_numpy(std::move(out));
      },
      "values"_a, "groups"_a.none(), "span"_a, "com1_update"_a = true);

  m.def(
      "linear_ar",
      [](const F64& target, const std::optional<I64>& groups, std::int64_t horizon, const U8& fit_mask,
         std::size_t lags, double alpha, bool clip_unit_interval) {
        std::vector<double> out;
        {
          const nb::gil_scoped_release release;
          out = paudit::linear_ar(span_of(target), groups_of(groups), horizon, span_of(fit_mask), lags, alpha,
                                  clip_unit_interval);
        }
        return to_numpy(std::move(out));
      },
      "target"_a, "groups"_a.none(), "horizon"_a, "fit_mask"_a, "lags"_a, "alpha"_a, "clip_unit_interval"_a);

  m.def(
      "estimate_season",
      [](const F64& target, const std::optional<I64>& groups, const U8& fit_mask, std::int64_t max_lag,
         std::int64_t min_lag) {
        const nb::gil_scoped_release release;
        return paudit::estimate_season(span_of(target), groups_of(groups), span_of(fit_mask), max_lag,
                                       min_lag);
      },
      "target"_a, "groups"_a.none(), "fit_mask"_a, "max_lag"_a = 2000, "min_lag"_a = 2);

  m.def(
      "optimal_threshold",
      [](const std::string& metric, const F64& y_true, const F64& y_score) {
        const auto mt = metric_of(metric);
        const nb::gil_scoped_release release;
        return paudit::optimal_threshold(mt, span_of(y_true), span_of(y_score));
      },
      "metric"_a, "y_true"_a, "y_score"_a);

  // --- full audit -----------------------------------------------------------------------------------
  nb::class_<AuditInputs>(m, "AuditInputs")
      .def(nb::init<>())
      .def_rw("target", &AuditInputs::target)
      .def_rw("pred", &AuditInputs::pred, nb::arg("value").none())
      .def_rw("group_keys", &AuditInputs::group_keys)
      .def_rw("time_i64", &AuditInputs::time_i64, nb::arg("value").none())
      .def_rw("time_f64", &AuditInputs::time_f64, nb::arg("value").none())
      .def_rw("split", &AuditInputs::split, nb::arg("value").none())
      .def_rw("feature_names", &AuditInputs::feature_names)
      .def_rw("features", &AuditInputs::features)
      .def_rw("extra_names", &AuditInputs::extra_names)
      .def_rw("extras", &AuditInputs::extras);

  m.def(
      "audit",
      [](const AuditInputs& inputs, const nb::dict& config) {
        const paudit::AuditColumns cols = columns_of(inputs);
        const paudit::AuditConfig cfg = config_of(config);
        paudit::AuditReport report;
        {
          const nb::gil_scoped_release release;
          report = paudit::audit(cols, cfg);
        }
        return report_to_py(std::move(report));
      },
      "inputs"_a, "config"_a, "Run the full audit natively; returns the report as a dict.");

  m.def(
      "audit_each_series",
      [](const AuditInputs& inputs, const nb::dict& config, const I64& series, std::size_t n_series) {
        const paudit::AuditColumns cols = columns_of(inputs);
        const paudit::AuditConfig cfg = config_of(config);
        std::vector<paudit::AuditReport> reports;
        {
          const nb::gil_scoped_release release;
          reports = paudit::audit_each_series(cols, cfg, span_of(series), n_series);
        }
        nb::list out;
        for (auto& r : reports) out.append(report_to_py(std::move(r)));
        return out;
      },
      "inputs"_a, "config"_a, "series"_a, "n_series"_a,
      "Audit every series in parallel (rows with series == -1 are skipped).");

  // --- test support -------------------------------------------------------------------------------
  m.def(
      "rng_integers",
      [](std::uint64_t seed, std::uint64_t high, std::size_t size) {
        paudit::rng::Pcg64 gen(seed);
        std::vector<std::int64_t> out(size);
        paudit::rng::integers(gen, high, out);
        return to_numpy(std::move(out));
      },
      "seed"_a, "high"_a, "size"_a,
      "np.random.default_rng(seed).integers(0, high, size), natively (for tests).");
}
