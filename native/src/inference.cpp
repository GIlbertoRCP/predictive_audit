#include "paudit/inference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>

#include "paudit/numeric.hpp"

namespace paudit {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

/// Stirling-series remainder of ln Gamma(z) for z >= 30.
double stirling_tail(double z) noexcept {
  const double z2 = z * z;
  return (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - (1.0 / 1680.0) / z2) / z2) / z2) / z;
}

/// ln B(a, b) without the catastrophic cancellation of lgamma differences for large arguments.
double log_beta(double a, double b) noexcept {
  if (a < b) std::swap(a, b);  // a >= b
  if (a >= 30.0) {
    // ln Gamma(a) - ln Gamma(a + b) by differencing Stirling's series analytically.
    const double ab = a + b;
    const double diff =
        -(a - 0.5) * std::log1p(b / a) - b * std::log(ab) + b + stirling_tail(a) - stirling_tail(ab);
    return std::lgamma(b) + diff;
  }
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

/// Continued fraction for I_x(a, b) (modified Lentz), valid for x < (a + 1) / (a + b + 2).
double beta_continued_fraction(double a, double b, double x) {
  constexpr double kTiny = 1e-300;
  constexpr double kEps = 1e-16;
  constexpr int kMaxIter = 200000;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < kTiny) d = kTiny;
  d = 1.0 / d;
  double h = d;
  for (int it = 1; it <= kMaxIter; ++it) {
    const auto m = static_cast<double>(it);
    const double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kTiny) d = kTiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < kTiny) c = kTiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < kEps) return h;
  }
  return h;  // converged to working precision in practice; never reached for df >= 1
}

std::optional<double> step_loss(Metric metric, double y, double m, double b, double thr) noexcept {
  switch (metric) {
    case Metric::mae:
      return std::abs(y - b) - std::abs(y - m);
    case Metric::rmse:
    case Metric::brier:
      return (y - b) * (y - b) - (y - m) * (y - m);
    case Metric::smape: {
      const double db = std::abs(y) + std::abs(b);
      const double dm = std::abs(y) + std::abs(m);
      const double lb = db == 0.0 ? 0.0 : 2.0 * std::abs(y - b) / db;
      const double lm = dm == 0.0 ? 0.0 : 2.0 * std::abs(y - m) / dm;
      return lb - lm;
    }
    case Metric::accuracy: {
      const bool t = y >= 0.5;
      return (t == (m >= thr) ? 1.0 : 0.0) - (t == (b >= thr) ? 1.0 : 0.0);
    }
    default:
      return std::nullopt;
  }
}

/// sum_i x[i] * x[i + lag] with eight independent accumulators. NumPy computes these with BLAS
/// (whose summation order is unspecified), so no particular order is needed for parity, and the
/// independent accumulators let the compiler vectorize.
double lagged_dot(const std::vector<double>& x, std::size_t lag) noexcept {
  const std::size_t len = x.size() - lag;
  const double* a = x.data();
  const double* b = x.data() + lag;
  std::array<double, 8> acc{};
  std::size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    for (std::size_t j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
  }
  double tail = 0.0;
  for (; i < len; ++i) tail += a[i] * b[i];
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
}

}  // namespace

double incomplete_beta(double a, double b, double x, double y) {
  if (!(a > 0.0) || !(b > 0.0)) throw std::invalid_argument("incomplete_beta: a and b must be positive");
  if (std::isnan(x) || std::isnan(y)) return kNaN;
  if (x <= 0.0) return 0.0;
  if (y <= 0.0) return 1.0;
  const double log_x = x > 0.5 ? std::log1p(-y) : std::log(x);
  const double log_y = y > 0.5 ? std::log1p(-x) : std::log(y);
  const double front = std::exp(a * log_x + b * log_y - log_beta(a, b));
  if (x < (a + 1.0) / (a + b + 2.0)) return front * beta_continued_fraction(a, b, x) / a;
  return 1.0 - front * beta_continued_fraction(b, a, y) / b;
}

double student_t_two_sided_p(double t, double df) {
  if (std::isnan(t) || std::isnan(df) || !(df > 0.0)) return kNaN;
  const double t2 = t * t;
  if (!std::isfinite(t2)) return 0.0;
  const double denom = df + t2;
  return incomplete_beta(0.5 * df, 0.5, df / denom, t2 / denom);
}

DmResult diebold_mariano(Metric metric, std::span<const double> y_true, std::span<const double> model,
                         std::span<const double> baseline, std::int64_t horizon, GroupCodes groups,
                         const TimeKeys& times, double threshold) {
  const std::size_t n_rows = y_true.size();
  if (model.size() != n_rows || baseline.size() != n_rows) {
    throw std::invalid_argument("y_true, model and baseline must have the same length");
  }
  if (!step_loss(metric, 0.0, 0.0, 0.0, threshold) || n_rows < 5) return {kNaN, kNaN};

  std::vector<double> d(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    d[i] = *step_loss(metric, y_true[i], model[i], baseline[i], threshold);
  }

  const PanelInfo panel = detect_aligned_panel(groups, times);
  if (panel.aligned) {
    std::vector<double> sums(panel.n_times, 0.0);
    std::vector<double> counts(panel.n_times, 0.0);
    for (std::size_t i = 0; i < n_rows; ++i) {
      const auto t = static_cast<std::size_t>(panel.time_index[i]);
      sums[t] += d[i];
      counts[t] += 1.0;
    }
    d.resize(panel.n_times);
    for (std::size_t t = 0; t < panel.n_times; ++t) d[t] = sums[t] / std::max(counts[t], 1.0);
  }

  const std::size_t n = d.size();
  if (n < 5) return {kNaN, kNaN};
  const auto nd = static_cast<double>(n);

  const double mean_d = numeric::mean(d);
  std::vector<double> dc(n);
  for (std::size_t i = 0; i < n; ++i) dc[i] = d[i] - mean_d;
  const double gamma0 = lagged_dot(dc, 0) / nd;
  if (gamma0 == 0.0) {
    if (mean_d == 0.0) return {0.0, 1.0};
    return {mean_d > 0.0 ? kInf : -kInf, 0.0};
  }

  const auto cube_root = static_cast<std::int64_t>(std::nearbyint(std::pow(nd, 1.0 / 3.0)));
  const std::int64_t max_lag = std::min<std::int64_t>(static_cast<std::int64_t>(n) - 1,
                                                      std::max<std::int64_t>(horizon - 1, cube_root));
  double lrv = gamma0;
  for (std::int64_t k = 1; k <= max_lag; ++k) {
    const double weight = 1.0 - static_cast<double>(k) / (static_cast<double>(max_lag) + 1.0);
    const auto ku = static_cast<std::size_t>(k);
    const double gamma_k = lagged_dot(dc, ku) / nd;
    lrv += 2.0 * weight * gamma_k;
  }
  if (lrv <= 0.0) lrv = gamma0;

  const double dm = mean_d / std::sqrt(lrv / nd);
  const auto h_eff = static_cast<double>(std::min<std::int64_t>(
      std::max<std::int64_t>(1, horizon), std::max<std::int64_t>(1, static_cast<std::int64_t>(n / 2))));
  const double hln = std::sqrt(std::max(1e-12, (nd + 1.0 - 2.0 * h_eff + h_eff * (h_eff - 1.0) / nd) / nd));
  const double stat = dm * hln;
  return {stat, student_t_two_sided_p(std::abs(stat), nd - 1.0)};
}

namespace {

std::pair<std::vector<std::size_t>, std::vector<double>> finite_sorted(std::span<const double> p) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (std::isfinite(p[i])) idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return p[a] < p[b]; });
  std::vector<double> ranked(idx.size());
  for (std::size_t r = 0; r < idx.size(); ++r) ranked[r] = p[idx[r]];
  return {std::move(idx), std::move(ranked)};
}

}  // namespace

std::vector<double> benjamini_hochberg(std::span<const double> p_values) {
  std::vector<double> q(p_values.size(), kNaN);
  auto [idx, ranked] = finite_sorted(p_values);
  const std::size_t m = ranked.size();
  if (m == 0) return q;
  const auto md = static_cast<double>(m);
  std::vector<double> adj(m);
  for (std::size_t r = 0; r < m; ++r) adj[r] = ranked[r] * md / static_cast<double>(r + 1);
  for (std::size_t r = m - 1; r-- > 0;) adj[r] = std::min(adj[r], adj[r + 1]);
  for (std::size_t r = 0; r < m; ++r) q[idx[r]] = std::clamp(adj[r], 0.0, 1.0);
  return q;
}

std::vector<double> holm_bonferroni(std::span<const double> p_values) {
  std::vector<double> out(p_values.size(), kNaN);
  auto [idx, ranked] = finite_sorted(p_values);
  const std::size_t m = ranked.size();
  if (m == 0) return out;
  std::vector<double> adj(m);
  for (std::size_t r = 0; r < m; ++r) adj[r] = ranked[r] * static_cast<double>(m - r);
  for (std::size_t r = 1; r < m; ++r) adj[r] = std::max(adj[r], adj[r - 1]);
  for (std::size_t r = 0; r < m; ++r) out[idx[r]] = std::clamp(adj[r], 0.0, 1.0);
  return out;
}

}  // namespace paudit
