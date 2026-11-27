#include "paudit/metrics.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include "paudit/numeric.hpp"
#include "paudit/rank.hpp"

namespace paudit {
namespace {

constexpr std::array<std::pair<std::string_view, Metric>, 11> kNames{{
    {"f1", Metric::f1},
    {"pr_auc", Metric::pr_auc},
    {"mcc", Metric::mcc},
    {"balanced_accuracy", Metric::balanced_accuracy},
    {"accuracy", Metric::accuracy},
    {"roc_auc", Metric::roc_auc},
    {"brier", Metric::brier},
    {"mae", Metric::mae},
    {"rmse", Metric::rmse},
    {"r2", Metric::r2},
    {"smape", Metric::smape},
}};

struct Confusion {
  double tp = 0.0;
  double fp = 0.0;
  double fn = 0.0;
  double tn = 0.0;
};

Confusion confusion(std::span<const double> yt, std::span<const double> yp, double threshold) {
  std::size_t tp = 0;
  std::size_t fp = 0;
  std::size_t fn = 0;
  std::size_t tn = 0;
  for (std::size_t i = 0; i < yt.size(); ++i) {
    const bool t = yt[i] >= 0.5;
    const bool p = yp[i] >= threshold;
    tp += static_cast<std::size_t>(t && p);
    fp += static_cast<std::size_t>(!t && p);
    fn += static_cast<std::size_t>(t && !p);
    tn += static_cast<std::size_t>(!t && !p);
  }
  return {static_cast<double>(tp), static_cast<double>(fp), static_cast<double>(fn), static_cast<double>(tn)};
}

double squared(double x) noexcept { return x * x; }

}  // namespace

std::optional<Metric> parse_metric(std::string_view name) noexcept {
  for (const auto& [n, m] : kNames) {
    if (n == name) return m;
  }
  return std::nullopt;
}

std::string_view metric_name(Metric m) noexcept {
  for (const auto& [n, v] : kNames) {
    if (v == m) return n;
  }
  return "unknown";
}

bool lower_is_better(Metric m) noexcept {
  return m == Metric::mae || m == Metric::rmse || m == Metric::smape || m == Metric::brier;
}

bool is_prefix_metric(Metric m) noexcept { return m != Metric::roc_auc && m != Metric::pr_auc; }

bool has_native_compute(Metric /*m*/) noexcept { return true; }

double compute(Metric m, std::span<const double> yt, std::span<const double> yp, double threshold) {
  if (yt.size() != yp.size()) throw std::invalid_argument("y_true and y_pred must have the same length");
  const std::size_t n = yt.size();
  const auto nd = static_cast<double>(n);
  switch (m) {
    case Metric::f1: {
      const Confusion c = confusion(yt, yp, threshold);
      const double denom = 2.0 * c.tp + c.fp + c.fn;
      return denom > 0.0 ? 2.0 * c.tp / denom : 0.0;
    }
    case Metric::accuracy: {
      std::size_t hits = 0;
      for (std::size_t i = 0; i < n; ++i)
        hits += static_cast<std::size_t>((yt[i] >= 0.5) == (yp[i] >= threshold));
      return static_cast<double>(hits) / nd;
    }
    case Metric::balanced_accuracy: {
      const Confusion c = confusion(yt, yp, threshold);
      const double pos = c.tp + c.fn;
      const double neg = c.tn + c.fp;
      if (pos > 0.0 && neg > 0.0) return 0.5 * (c.tp / pos + c.tn / neg);
      if (pos > 0.0) return c.tp / pos;
      return neg > 0.0 ? c.tn / neg : 0.0;
    }
    case Metric::mcc: {
      const Confusion c = confusion(yt, yp, threshold);
      const double denom = std::sqrt((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn));
      return denom > 0.0 ? (c.tp * c.tn - c.fp * c.fn) / denom : 0.0;
    }
    case Metric::brier:
      return numeric::pairwise_sum(n, [&](std::size_t i) { return squared(yt[i] - yp[i]); }) / nd;
    case Metric::mae:
      return numeric::pairwise_sum(n, [&](std::size_t i) { return std::abs(yt[i] - yp[i]); }) / nd;
    case Metric::rmse:
      return std::sqrt(numeric::pairwise_sum(n, [&](std::size_t i) { return squared(yt[i] - yp[i]); }) / nd);
    case Metric::r2: {
      const double sse = numeric::pairwise_sum(n, [&](std::size_t i) { return squared(yt[i] - yp[i]); });
      const double mu = numeric::mean(yt);
      const double sst = numeric::pairwise_sum(n, [&](std::size_t i) { return squared(yt[i] - mu); });
      if (sst == 0.0) return sse == 0.0 ? 1.0 : 0.0;
      return 1.0 - sse / sst;
    }
    case Metric::smape:
      return numeric::pairwise_sum(n,
                                   [&](std::size_t i) {
                                     const double d = std::abs(yt[i]) + std::abs(yp[i]);
                                     return d == 0.0 ? 0.0 : 2.0 * std::abs(yt[i] - yp[i]) / d;
                                   }) /
             nd;
    case Metric::roc_auc:
      return roc_auc(yt, yp);
    case Metric::pr_auc:
      return average_precision(yt, yp);
  }
  throw std::invalid_argument("metric " + std::string(metric_name(m)) + " is not implemented natively");
}

double score(Metric m, std::span<const double> yt, std::span<const double> yp, double threshold) {
  const double v = compute(m, yt, yp, threshold);
  return lower_is_better(m) ? -v : v;
}

}  // namespace paudit
