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

#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paudit/bootstrap.hpp"
#include "paudit/inference.hpp"
#include "paudit/metrics.hpp"
#include "paudit/parallel.hpp"
#include "paudit/rng.hpp"
#include "paudit/version.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using F64 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64 = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using NumpyOut = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

std::span<const double> span_of(const F64& a) { return {a.data(), a.shape(0)}; }
std::span<const std::int64_t> span_of(const I64& a) { return {a.data(), a.shape(0)}; }

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
