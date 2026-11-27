#include "paudit/bootstrap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "paudit/numeric.hpp"
#include "paudit/parallel.hpp"
#include "paudit/rng.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wnull-dereference"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define POCKETFFT_NO_MULTITHREADING
#include <pocketfft_hdronly.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace paudit {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// --- np.unique(..., return_inverse=True) ---------------------------------------------------

template <class T>
bool is_nan(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(v);
  } else {
    (void)v;
    return false;
  }
}

/// Dense codes of sorted unique values; NaNs sort last and collapse into one value (NumPy >= 1.21).
template <class T>
std::vector<std::int64_t> unique_inverse(std::span<const T> v, std::size_t& n_unique) {
  const std::size_t n = v.size();
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const bool na = is_nan(v[a]);
    const bool nb = is_nan(v[b]);
    if (na || nb) return !na && nb;
    return v[a] < v[b];
  });
  std::vector<std::int64_t> inv(n);
  std::int64_t code = -1;
  for (std::size_t r = 0; r < n; ++r) {
    const std::size_t i = order[r];
    bool same = false;
    if (r > 0) {
      const std::size_t p = order[r - 1];
      same = (is_nan(v[i]) && is_nan(v[p])) || (!is_nan(v[i]) && !is_nan(v[p]) && v[i] == v[p]);
    }
    if (!same) ++code;
    inv[i] = code;
  }
  n_unique = static_cast<std::size_t>(code + 1);
  return inv;
}

std::size_t times_size(const TimeKeys& times) {
  if (const auto* ti = std::get_if<std::span<const std::int64_t>>(&times)) return ti->size();
  if (const auto* tf = std::get_if<std::span<const double>>(&times)) return tf->size();
  return 0;
}

// --- per-metric sufficient statistics and evaluation ------------------------------------------

struct Inputs {
  std::span<const double> y;
  std::span<const double> m;
  std::span<const double> b;
  double threshold;
};

template <std::size_t K>
using Row = std::array<double, K>;

double sq(double x) noexcept { return x * x; }

double smape_term(double y, double p) noexcept {
  const double d = std::abs(y) + std::abs(p);
  return d == 0.0 ? 0.0 : 2.0 * std::abs(y - p) / d;
}

void confusion_row(double y, double p, double thr, double* out) noexcept {
  const bool t = y >= 0.5;
  const bool q = p >= thr;
  out[0] = (t && q) ? 1.0 : 0.0;
  out[1] = (!t && q) ? 1.0 : 0.0;
  out[2] = (t && !q) ? 1.0 : 0.0;
  out[3] = (!t && !q) ? 1.0 : 0.0;
}

double f1_from(double tp, double fp, double fn) noexcept {
  const double d = 2.0 * tp + fp + fn;
  return d > 0.0 ? 2.0 * tp / d : 0.0;
}

double balanced_from(double tp, double fp, double fn, double tn) noexcept {
  const double pos = tp + fn;
  const double neg = tn + fp;
  const double tpr = pos > 0.0 ? tp / pos : 0.0;
  const double tnr = neg > 0.0 ? tn / neg : 0.0;
  if (pos > 0.0 && neg > 0.0) return 0.5 * (tpr + tnr);
  if (pos > 0.0 && neg == 0.0) return tpr;
  return tnr;
}

double mcc_from(double tp, double fp, double fn, double tn) noexcept {
  const double d = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return d > 0.0 ? (tp * tn - fp * fn) / d : 0.0;
}

double r2_from(double e, double sst) noexcept {
  if (sst == 0.0) return e == 0.0 ? 1.0 : 0.0;
  return 1.0 - e / sst;
}

// --- generic prefix-sum circular block bootstrap ----------------------------------------------

constexpr std::size_t kMaxStartsPerChunk = std::size_t{1} << 22;  // 32 MiB of int64 block starts
constexpr std::size_t kLookupsPerTask = std::size_t{1} << 15;

/// Sum of KC interleaved prefix columns over the draw's blocks, in NumPy's evaluation order.
template <std::size_t KC>
void block_sums(const double* prefix, const std::int64_t* starts, std::size_t n_blocks, std::size_t b_len,
                std::size_t rem, Row<KC>& out) {
  if (n_blocks == 1) {
    const auto s0 = static_cast<std::size_t>(starts[0]);
    for (std::size_t k = 0; k < KC; ++k) out[k] = prefix[(s0 + rem) * KC + k] - prefix[s0 * KC + k];
    return;
  }
  auto term = [&](std::size_t j, Row<KC>& v) {
    const auto s = static_cast<std::size_t>(starts[j]);
    const double* lo = prefix + s * KC;
    const double* hi = prefix + (s + b_len) * KC;
    for (std::size_t k = 0; k < KC; ++k) v[k] = hi[k] - lo[k];
  };
  numeric::pairwise_sum_k<KC>(term, 0, n_blocks - 1, out);
  const auto last = static_cast<std::size_t>(starts[n_blocks - 1]);
  for (std::size_t k = 0; k < KC; ++k)
    out[k] = out[k] + (prefix[(last + rem) * KC + k] - prefix[last * KC + k]);
}

/// Interleaved exclusive prefix sums over `length` padded positions, `value(j, row)` per position.
template <std::size_t KC, class Value>
std::vector<double> interleaved_prefix(std::size_t length, Value value) {
  std::vector<double> prefix((length + 1) * KC, 0.0);
  Row<KC> acc{};
  Row<KC> v{};
  for (std::size_t j = 0; j < length; ++j) {
    value(j, v);
    for (std::size_t k = 0; k < KC; ++k) {
      acc[k] += v[k];
      prefix[(j + 1) * KC + k] = acc[k];
    }
  }
  return prefix;
}

template <std::size_t K, class RowStats, class Eval>
Interval bootstrap_impl(std::size_t n, RowStats row_stats, Eval eval, std::int64_t block, std::size_t n_boot,
                        std::uint64_t seed, GroupCodes groups, const PanelInfo& panel) {
  rng::Pcg64 gen(seed);
  std::vector<double> diffs(n_boot, kNaN);

  std::vector<double> rows(n * K);
  for (std::size_t i = 0; i < n; ++i) {
    Row<K> r{};
    row_stats(i, r);
    std::copy(r.begin(), r.end(), rows.begin() + static_cast<std::ptrdiff_t>(i * K));
  }

  auto clamp_block = [block](std::size_t len) {
    const auto b = std::min<std::int64_t>(block, static_cast<std::int64_t>(len));
    return static_cast<std::size_t>(std::max<std::int64_t>(1, b));
  };

  auto run_draws = [&](std::size_t high, const std::vector<std::int64_t>* remap, std::size_t n_blocks,
                       auto&& eval_draw) {
    const std::size_t chunk = std::max<std::size_t>(1, std::min(n_boot, kMaxStartsPerChunk / n_blocks));
    std::vector<std::int64_t> starts(chunk * n_blocks);
    const std::size_t grain = std::max<std::size_t>(1, kLookupsPerTask / n_blocks);
    for (std::size_t i0 = 0; i0 < n_boot; i0 += chunk) {
      const std::size_t sz = std::min(chunk, n_boot - i0);
      const std::span<std::int64_t> draw_starts(starts.data(), sz * n_blocks);
      rng::integers(gen, high, draw_starts);
      if (remap != nullptr) {
        for (auto& s : draw_starts) s = (*remap)[static_cast<std::size_t>(s)];
      }
      parallel::parallel_for(sz, grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t d = begin; d < end; ++d) diffs[i0 + d] = eval_draw(starts.data() + d * n_blocks);
      });
    }
  };

  if (panel.aligned) {
    constexpr std::size_t KC = K + 1;  // last column: row counts per timestamp
    const std::size_t eff_n = panel.n_times;
    const std::size_t b_len = clamp_block(eff_n);
    const std::size_t pad = b_len - 1;
    std::vector<double> agg(eff_n * KC, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const auto t = static_cast<std::size_t>(panel.time_index[i]);
      for (std::size_t k = 0; k < K; ++k) agg[t * KC + k] += rows[i * K + k];
      agg[t * KC + K] += 1.0;
    }
    const auto prefix = interleaved_prefix<KC>(eff_n + pad, [&](std::size_t j, Row<KC>& v) {
      const std::size_t t = j % eff_n;
      for (std::size_t k = 0; k < KC; ++k) v[k] = agg[t * KC + k];
    });
    const std::size_t n_blocks = (eff_n + b_len - 1) / b_len;
    const std::size_t rem = eff_n - (n_blocks - 1) * b_len;
    run_draws(eff_n, nullptr, n_blocks, [&](const std::int64_t* s) {
      Row<KC> sums{};
      block_sums<KC>(prefix.data(), s, n_blocks, b_len, rem, sums);
      Row<K> stats{};
      std::copy_n(sums.begin(), K, stats.begin());
      return eval(stats, sums[K]);
    });
  } else {
    const std::size_t b_len = clamp_block(n);
    const std::size_t pad = b_len - 1;
    const std::size_t n_blocks = (n + b_len - 1) / b_len;
    const std::size_t rem = n - (n_blocks - 1) * b_len;
    const auto denom = static_cast<double>(n);
    std::vector<double> prefix;
    std::vector<std::int64_t> valid_starts;
    if (groups) {
      CircularLayout layout = group_circular_layout(*groups, pad);
      prefix = interleaved_prefix<K>(layout.gather.size(), [&](std::size_t j, Row<K>& v) {
        const auto i = static_cast<std::size_t>(layout.gather[j]);
        for (std::size_t k = 0; k < K; ++k) v[k] = rows[i * K + k];
      });
      valid_starts = std::move(layout.starts);
    } else {
      prefix = interleaved_prefix<K>(n + pad, [&](std::size_t j, Row<K>& v) {
        const std::size_t i = j % n;
        for (std::size_t k = 0; k < K; ++k) v[k] = rows[i * K + k];
      });
    }
    const std::size_t high = groups ? valid_starts.size() : n;
    run_draws(high, groups ? &valid_starts : nullptr, n_blocks, [&](const std::int64_t* s) {
      Row<K> sums{};
      block_sums<K>(prefix.data(), s, n_blocks, b_len, rem, sums);
      return eval(sums, denom);
    });
  }

  std::vector<double> finite;
  finite.reserve(n_boot);
  for (const double d : diffs) {
    if (!std::isnan(d)) finite.push_back(d);
  }
  if (finite.empty()) return {kNaN, kNaN};
  std::vector<double> copy = finite;
  const double lo = numeric::percentile_linear(finite, 2.5);
  const double hi = numeric::percentile_linear(copy, 97.5);
  return {lo, hi};
}

}  // namespace

// --- panel structure ----------------------------------------------------------------------------

PanelInfo detect_aligned_panel(GroupCodes groups, const TimeKeys& times) {
  if (!groups || std::holds_alternative<std::monostate>(times) || groups->size() < 2) return {};
  const auto g = *groups;
  const std::size_t n = g.size();
  if (times_size(times) != n) throw std::invalid_argument("groups and times must have the same length");
  if (std::all_of(g.begin(), g.end(), [&](std::int64_t v) { return v == g[0]; })) return {};
  const auto nd = static_cast<double>(n);

  if (const auto* ti = std::get_if<std::span<const std::int64_t>>(&times)) {
    const auto [mn, mx] = std::minmax_element(ti->begin(), ti->end());
    const std::uint64_t diff = static_cast<std::uint64_t>(*mx) - static_cast<std::uint64_t>(*mn);
    if (diff < n) {
      const std::size_t span = static_cast<std::size_t>(diff) + 1;
      if (span >= 10 && nd / static_cast<double>(span) >= 1.5) {
        std::vector<std::int64_t> inv(n);
        std::vector<char> present(span, 0);
        for (std::size_t i = 0; i < n; ++i) {
          inv[i] = (*ti)[i] - *mn;
          present[static_cast<std::size_t>(inv[i])] = 1;
        }
        if (std::all_of(present.begin(), present.end(), [](char c) { return c != 0; })) {
          return {true, std::move(inv), span};
        }
      }
    }
  }

  std::size_t n_times = 0;
  std::vector<std::int64_t> inv;
  if (const auto* ti = std::get_if<std::span<const std::int64_t>>(&times)) {
    inv = unique_inverse(*ti, n_times);
  } else {
    inv = unique_inverse(std::get<std::span<const double>>(times), n_times);
  }
  if (n_times >= 10 && nd / static_cast<double>(n_times) >= 1.5) return {true, std::move(inv), n_times};
  return {};
}

CircularLayout group_circular_layout(std::span<const std::int64_t> groups, std::size_t pad) {
  CircularLayout out;
  const std::size_t n = groups.size();
  if (n == 0) return out;
  out.gather.reserve(n + pad);
  out.starts.reserve(n);
  std::size_t offset = 0;
  std::size_t s = 0;
  while (s < n) {
    std::size_t e = s + 1;
    while (e < n && groups[e] == groups[e - 1]) ++e;
    const std::size_t length = e - s;
    for (std::size_t j = 0; j < length + pad; ++j)
      out.gather.push_back(static_cast<std::int64_t>(s + j % length));
    for (std::size_t j = 0; j < length; ++j) out.starts.push_back(static_cast<std::int64_t>(offset + j));
    offset += length + pad;
    s = e;
  }
  return out;
}

std::size_t default_block_length(std::size_t n_total, GroupCodes groups, const TimeKeys& times) {
  const PanelInfo panel = detect_aligned_panel(groups, times);
  std::size_t t_eff = n_total;
  if (panel.aligned) {
    t_eff = panel.n_times;
  } else if (groups && !groups->empty()) {
    std::vector<std::size_t> lengths;
    const auto g = *groups;
    std::size_t s = 0;
    for (std::size_t i = 1; i <= g.size(); ++i) {
      if (i == g.size() || g[i] != g[i - 1]) {
        lengths.push_back(i - s);
        s = i;
      }
    }
    std::sort(lengths.begin(), lengths.end());
    const std::size_t m = lengths.size();
    const double median =
        (m % 2 == 1) ? static_cast<double>(lengths[m / 2])
                     : (static_cast<double>(lengths[m / 2 - 1]) + static_cast<double>(lengths[m / 2])) / 2.0;
    t_eff = static_cast<std::size_t>(median);
  }
  const double root = std::sqrt(static_cast<double>(std::max<std::size_t>(1, t_eff)));
  const std::size_t raw = std::max<std::size_t>(10, static_cast<std::size_t>(std::nearbyint(root)));
  if (t_eff >= 20) return std::max<std::size_t>(1, std::min(raw, std::max<std::size_t>(1, t_eff / 2)));
  return std::max<std::size_t>(1, std::min(raw, t_eff));
}

std::size_t estimate_block_length(std::span<const double> x, std::optional<std::size_t> max_block) {
  std::vector<double> arr;
  arr.reserve(x.size());
  for (const double v : x) {
    if (std::isfinite(v)) arr.push_back(v);
  }
  const std::size_t n = arr.size();
  const std::size_t cap = max_block ? *max_block : std::max<std::size_t>(1, n / 2);
  const std::size_t small = std::max<std::size_t>(1, std::min<std::size_t>(10, cap));
  if (n < 20) return small;
  const auto nd = static_cast<double>(n);

  const double mu = numeric::mean(arr);
  std::vector<double> xc(n);
  for (std::size_t i = 0; i < n; ++i) xc[i] = arr[i] - mu;
  const double var = numeric::pairwise_sum(n, [&](std::size_t i) { return xc[i] * xc[i]; }) / nd;
  if (var <= 0.0) return small;

  // O(n log n) autocovariance: irfft(|rfft(xc, n_fft)|^2, n_fft)[:n] / n, as NumPy computes it.
  const auto n_fft = std::size_t{1} << static_cast<unsigned>(std::ceil(std::log2(2.0 * nd)));
  std::vector<double> padded(n_fft, 0.0);
  std::copy(xc.begin(), xc.end(), padded.begin());
  std::vector<std::complex<double>> spectrum(n_fft / 2 + 1);
  const pocketfft::shape_t shape{n_fft};
  const pocketfft::stride_t real_stride{static_cast<std::ptrdiff_t>(sizeof(double))};
  const pocketfft::stride_t complex_stride{static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
  pocketfft::r2c(shape, real_stride, complex_stride, 0, pocketfft::FORWARD, padded.data(), spectrum.data(),
                 1.0);
  for (auto& c : spectrum) c = std::complex<double>(c.real() * c.real() + c.imag() * c.imag(), 0.0);
  std::vector<double> circ(n_fft);
  pocketfft::c2r(shape, complex_stride, real_stride, 0, pocketfft::BACKWARD, spectrum.data(), circ.data(),
                 1.0 / static_cast<double>(n_fft));
  std::vector<double> acov(n);
  for (std::size_t i = 0; i < n; ++i) acov[i] = circ[i] / nd;

  const double log10n = std::log10(nd);
  const auto kn = std::max<std::size_t>(5, static_cast<std::size_t>(std::ceil(std::sqrt(log10n))));
  const double thresh = 2.0 * std::sqrt(log10n / nd);
  const std::size_t max_m =
      std::min(n / 2, std::max<std::size_t>(10, static_cast<std::size_t>(std::ceil(std::sqrt(nd))) + kn));
  std::size_t m_hat = max_m;
  for (std::size_t lag = 1; lag + kn <= max_m; ++lag) {
    bool quiet = true;
    for (std::size_t j = lag; j < lag + kn; ++j) {
      if (!(std::abs(acov[j] / var) < thresh)) {
        quiet = false;
        break;
      }
    }
    if (quiet) {
      m_hat = std::max<std::size_t>(1, lag);
      break;
    }
  }
  const std::size_t bw = std::min(2 * m_hat, n / 2);
  if (bw < 1) return 1;

  const auto bwd = static_cast<double>(bw);
  auto weight = [bwd](std::size_t lag) {
    const double u = static_cast<double>(lag) / bwd;
    return u <= 0.5 ? 1.0 : 2.0 * (1.0 - u);
  };
  const double g_hat = 2.0 * numeric::pairwise_sum(bw, [&](std::size_t i) {
                         const std::size_t lag = i + 1;
                         return weight(lag) * static_cast<double>(lag) * acov[lag];
                       });
  const double g0_hat =
      acov[0] + 2.0 * numeric::pairwise_sum(bw, [&](std::size_t i) { return weight(i + 1) * acov[i + 1]; });
  if (g0_hat <= 0.0 || g_hat == 0.0) {
    return std::max<std::size_t>(1, std::min(static_cast<std::size_t>(std::nearbyint(std::sqrt(nd))), cap));
  }
  const double d_cbb = (4.0 / 3.0) * (g0_hat * g0_hat);
  const double b_opt = std::pow((2.0 * (g_hat * g_hat)) / d_cbb, 1.0 / 3.0) * std::pow(nd, 1.0 / 3.0);
  return std::max<std::size_t>(1, std::min(static_cast<std::size_t>(std::ceil(b_opt)), cap));
}

// --- dispatcher -----------------------------------------------------------------------------------

Interval block_bootstrap_diff(Metric metric, std::span<const double> y_true, std::span<const double> model,
                              std::span<const double> baseline, std::int64_t block, std::int64_t n_boot,
                              std::uint64_t seed, GroupCodes groups, const TimeKeys& times,
                              double threshold) {
  const std::size_t n = y_true.size();
  if (model.size() != n || baseline.size() != n) {
    throw std::invalid_argument("y_true, model and baseline must have the same length");
  }
  if (groups && groups->size() != n) throw std::invalid_argument("groups must have one entry per row");
  if (!std::holds_alternative<std::monostate>(times) && times_size(times) != n) {
    throw std::invalid_argument("times must have one entry per row");
  }
  if (!is_prefix_metric(metric)) {
    throw std::invalid_argument("rank-metric bootstrap is not implemented natively yet");
  }
  if (n == 0 || n_boot <= 0) return {kNaN, kNaN};

  const PanelInfo panel = detect_aligned_panel(groups, times);
  const Inputs in{y_true, model, baseline, threshold};
  const auto nb = static_cast<std::size_t>(n_boot);

  switch (metric) {
    case Metric::mae:
      return bootstrap_impl<2>(
          n,
          [&](std::size_t i, Row<2>& r) { r = {std::abs(in.y[i] - in.m[i]), std::abs(in.y[i] - in.b[i])}; },
          [](const Row<2>& s, double d) { return (s[1] - s[0]) / d; }, block, nb, seed, groups, panel);
    case Metric::rmse:
    case Metric::brier: {
      auto rows = [&](std::size_t i, Row<2>& r) { r = {sq(in.y[i] - in.m[i]), sq(in.y[i] - in.b[i])}; };
      if (metric == Metric::rmse) {
        return bootstrap_impl<2>(
            n, rows, [](const Row<2>& s, double d) { return std::sqrt(s[1] / d) - std::sqrt(s[0] / d); },
            block, nb, seed, groups, panel);
      }
      return bootstrap_impl<2>(
          n, rows, [](const Row<2>& s, double d) { return (s[1] - s[0]) / d; }, block, nb, seed, groups,
          panel);
    }
    case Metric::smape:
      return bootstrap_impl<2>(
          n,
          [&](std::size_t i, Row<2>& r) { r = {smape_term(in.y[i], in.m[i]), smape_term(in.y[i], in.b[i])}; },
          [](const Row<2>& s, double d) { return (s[1] - s[0]) / d; }, block, nb, seed, groups, panel);
    case Metric::r2:
      return bootstrap_impl<4>(
          n,
          [&](std::size_t i, Row<4>& r) {
            r = {sq(in.y[i] - in.m[i]), sq(in.y[i] - in.b[i]), in.y[i], sq(in.y[i])};
          },
          [](const Row<4>& s, double d) {
            const double sst = s[3] - sq(s[2]) / d;
            return r2_from(s[0], sst) - r2_from(s[1], sst);
          },
          block, nb, seed, groups, panel);
    case Metric::accuracy:
      return bootstrap_impl<2>(
          n,
          [&](std::size_t i, Row<2>& r) {
            const bool t = in.y[i] >= 0.5;
            r = {t == (in.m[i] >= in.threshold) ? 1.0 : 0.0, t == (in.b[i] >= in.threshold) ? 1.0 : 0.0};
          },
          [](const Row<2>& s, double d) { return (s[0] - s[1]) / d; }, block, nb, seed, groups, panel);
    case Metric::f1:
    case Metric::mcc:
    case Metric::balanced_accuracy: {
      auto rows = [&](std::size_t i, Row<8>& r) {
        confusion_row(in.y[i], in.m[i], in.threshold, r.data());
        confusion_row(in.y[i], in.b[i], in.threshold, r.data() + 4);
      };
      if (metric == Metric::f1) {
        return bootstrap_impl<8>(
            n, rows,
            [](const Row<8>& s, double) { return f1_from(s[0], s[1], s[2]) - f1_from(s[4], s[5], s[6]); },
            block, nb, seed, groups, panel);
      }
      if (metric == Metric::mcc) {
        return bootstrap_impl<8>(
            n, rows,
            [](const Row<8>& s, double) {
              return mcc_from(s[0], s[1], s[2], s[3]) - mcc_from(s[4], s[5], s[6], s[7]);
            },
            block, nb, seed, groups, panel);
      }
      return bootstrap_impl<8>(
          n, rows,
          [](const Row<8>& s, double) {
            return balanced_from(s[0], s[1], s[2], s[3]) - balanced_from(s[4], s[5], s[6], s[7]);
          },
          block, nb, seed, groups, panel);
    }
    case Metric::roc_auc:
    case Metric::pr_auc:
      break;
  }
  throw std::invalid_argument("unsupported bootstrap metric");
}

}  // namespace paudit
