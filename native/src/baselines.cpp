#include "paudit/baselines.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "fft.hpp"
#include "paudit/numeric.hpp"
#include "paudit/parallel.hpp"

namespace paudit {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// pandas window functions treat +-inf as missing (BaseWindow._prep_values).
double finite_or_nan(double v) noexcept { return std::isinf(v) ? kNaN : v; }

/// Run `fn(group_index, first_row_slot, last_row_slot)` for every group, in parallel when worthwhile.
template <class Fn>
void for_each_group(const GroupRows& g, Fn&& fn) {
  const std::size_t n_groups = g.n_groups();
  const std::size_t grain = n_groups > 64 ? 8 : n_groups;
  parallel::parallel_for(n_groups, std::max<std::size_t>(1, grain), [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) fn(g.offsets[k], g.offsets[k + 1]);
  });
}

// --- pandas roll_mean state (aggregations.pyx: add_mean / remove_mean / calc_mean) ----------

struct RollMean {
  double sum_x = 0.0;
  double compensation_add = 0.0;
  double compensation_remove = 0.0;
  double prev_value = 0.0;
  std::int64_t num_consecutive_same_value = 0;
  std::int64_t nobs = 0;
  std::int64_t neg_ct = 0;

  void reset(double first) noexcept {
    sum_x = compensation_add = compensation_remove = 0.0;
    nobs = neg_ct = 0;
    prev_value = first;
    num_consecutive_same_value = 0;
  }

  void add(double val) noexcept {
    if (val != val) return;
    ++nobs;
    const double y = val - compensation_add;
    const double t = sum_x + y;
    compensation_add = t - sum_x - y;
    sum_x = t;
    if (std::signbit(val)) ++neg_ct;
    if (val == prev_value) {
      ++num_consecutive_same_value;
    } else {
      num_consecutive_same_value = 1;
    }
    prev_value = val;
  }

  void remove(double val) noexcept {
    if (val != val) return;
    --nobs;
    const double y = -val - compensation_remove;
    const double t = sum_x + y;
    compensation_remove = t - sum_x - y;
    sum_x = t;
    if (std::signbit(val)) --neg_ct;
  }

  [[nodiscard]] double mean(std::int64_t minp) const noexcept {
    if (!(nobs >= minp && nobs > 0)) return kNaN;
    double result = sum_x / static_cast<double>(nobs);
    if (num_consecutive_same_value >= nobs) {
      result = prev_value;
    } else if (neg_ct == 0 && result < 0) {
      result = 0;
    } else if (neg_ct == nobs && result > 0) {
      result = 0;
    }
    return result;
  }
};

// --- small dense linear algebra for the Ridge AR baseline ------------------------------------

/// Solve A x = b (A row-major d x d) by Gaussian elimination with partial pivoting, the method
/// LAPACK's dgesv uses. Throws on an exactly singular pivot, where NumPy raises LinAlgError.
std::vector<double> solve(std::vector<double> a, std::vector<double> b, std::size_t d) {
  for (std::size_t col = 0; col < d; ++col) {
    std::size_t piv = col;
    for (std::size_t r = col + 1; r < d; ++r) {
      if (std::abs(a[r * d + col]) > std::abs(a[piv * d + col])) piv = r;
    }
    if (a[piv * d + col] == 0.0) throw std::runtime_error("singular matrix");
    if (piv != col) {
      for (std::size_t c = 0; c < d; ++c) std::swap(a[col * d + c], a[piv * d + c]);
      std::swap(b[col], b[piv]);
    }
    for (std::size_t r = col + 1; r < d; ++r) {
      const double f = a[r * d + col] / a[col * d + col];
      for (std::size_t c = col; c < d; ++c) a[r * d + c] -= f * a[col * d + c];
      b[r] -= f * b[col];
    }
  }
  std::vector<double> x(d);
  for (std::size_t i = d; i-- > 0;) {
    double s = b[i];
    for (std::size_t c = i + 1; c < d; ++c) s -= a[i * d + c] * x[c];
    x[i] = s / a[i * d + i];
  }
  return x;
}

/// Closed-form Ridge with an unpenalized intercept (last column) over the selected rows.
std::vector<double> fit_ridge(const std::vector<double>& x, std::span<const double> y, std::size_t d,
                              const std::vector<std::size_t>& rows, double alpha) {
  std::vector<double> xtx(d * d, 0.0);
  std::vector<double> xty(d, 0.0);
  for (const std::size_t r : rows) {
    const double* xr = x.data() + r * d;
    for (std::size_t i = 0; i < d; ++i) {
      xty[i] += xr[i] * y[r];
      for (std::size_t j = 0; j < d; ++j) xtx[i * d + j] += xr[i] * xr[j];
    }
  }
  for (std::size_t i = 0; i + 1 < d; ++i) xtx[i * d + i] += alpha;
  return solve(std::move(xtx), std::move(xty), d);
}

}  // namespace

GroupRows group_rows(std::size_t n, GroupCodes groups) {
  GroupRows out;
  if (!groups) {
    out.rows.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.rows[i] = i;
    out.offsets = {0, n};
    return out;
  }
  const auto g = *groups;
  if (g.size() != n) throw std::invalid_argument("groups must have one entry per row");
  std::unordered_map<std::int64_t, std::size_t> slot;
  std::vector<std::size_t> code(n);
  std::vector<std::size_t> counts;
  for (std::size_t i = 0; i < n; ++i) {
    const auto [it, inserted] = slot.try_emplace(g[i], counts.size());
    if (inserted) counts.push_back(0);
    code[i] = it->second;
    ++counts[it->second];
  }
  out.offsets.assign(counts.size() + 1, 0);
  for (std::size_t k = 0; k < counts.size(); ++k) out.offsets[k + 1] = out.offsets[k] + counts[k];
  std::vector<std::size_t> fill(out.offsets.begin(), out.offsets.end() - 1);
  out.rows.resize(n);
  for (std::size_t i = 0; i < n; ++i) out.rows[fill[code[i]]++] = i;
  return out;
}

std::vector<double> lag(std::span<const double> values, GroupCodes groups, std::int64_t k) {
  if (k < 0) throw std::invalid_argument("lag must be >= 0");
  const auto ku = static_cast<std::size_t>(k);
  const GroupRows g = group_rows(values.size(), groups);
  std::vector<double> out(values.size(), kNaN);
  for_each_group(g, [&](std::size_t b, std::size_t e) {
    for (std::size_t p = b + ku; p < e; ++p) out[g.rows[p]] = values[g.rows[p - ku]];
  });
  return out;
}

std::vector<double> rolling_mean(std::span<const double> values, GroupCodes groups, std::size_t window,
                                 std::size_t min_periods) {
  if (window == 0) throw std::invalid_argument("window must be >= 1");
  const GroupRows g = group_rows(values.size(), groups);
  std::vector<double> out(values.size(), kNaN);
  const auto minp = static_cast<std::int64_t>(min_periods);
  for_each_group(g, [&](std::size_t b, std::size_t e) {
    // pandas FixedWindowIndexer per group: window i covers local [max(0, i + 1 - w), i + 1).
    auto v = [&](std::size_t local) { return finite_or_nan(values[g.rows[b + local]]); };
    RollMean st;
    std::size_t prev_s = 0;
    std::size_t prev_e = 0;
    for (std::size_t i = 0; i < e - b; ++i) {
      const std::size_t s = i + 1 >= window ? i + 1 - window : 0;
      const std::size_t en = i + 1;
      if (i == 0 || s >= prev_e) {
        st.reset(v(s));
        for (std::size_t j = s; j < en; ++j) st.add(v(j));
      } else {
        for (std::size_t j = prev_s; j < s; ++j) st.remove(v(j));
        for (std::size_t j = prev_e; j < en; ++j) st.add(v(j));
      }
      out[g.rows[b + i]] = st.mean(minp);
      prev_s = s;
      prev_e = en;
    }
  });
  return out;
}

std::vector<double> ewm_mean(std::span<const double> values, GroupCodes groups, double span,
                             bool com1_update) {
  if (!(span >= 1.0)) throw std::invalid_argument("span must be >= 1");
  const GroupRows g = group_rows(values.size(), groups);
  std::vector<double> out(values.size(), kNaN);
  const double com = (span - 1.0) / 2.0;
  const double alpha = 1.0 / (1.0 + com);
  const double old_wt_factor = 1.0 - alpha;
  constexpr std::int64_t kMinp = 1;
  for_each_group(g, [&](std::size_t b, std::size_t e) {
    // pandas aggregations.ewm with adjust=False, ignore_na=False, normalize=True, no deltas.
    double new_wt = alpha;
    double weighted = finite_or_nan(values[g.rows[b]]);
    std::int64_t nobs = weighted == weighted ? 1 : 0;
    out[g.rows[b]] = nobs >= kMinp ? weighted : kNaN;
    double old_wt = 1.0;
    for (std::size_t p = b + 1; p < e; ++p) {
      const double cur = finite_or_nan(values[g.rows[p]]);
      const bool is_observation = cur == cur;
      nobs += is_observation ? 1 : 0;
      if (weighted == weighted) {
        old_wt *= old_wt_factor;  // is_observation or not ignore_na (always true here)
        if (is_observation) {
          if (weighted != cur) {
            if (com1_update && com == 1.0) new_wt = 1.0 - old_wt;  // pandas >= 3.0: not adjust and com == 1
            weighted = old_wt * weighted + new_wt * cur;
            weighted /= (old_wt + new_wt);
          }
          old_wt = 1.0;
        }
      } else if (is_observation) {
        weighted = cur;
      }
      out[g.rows[p]] = nobs >= kMinp ? weighted : kNaN;
    }
  });
  return out;
}

std::vector<double> linear_ar(std::span<const double> target, GroupCodes groups, std::int64_t horizon,
                              std::span<const std::uint8_t> fit_mask, std::size_t lags, double alpha,
                              bool clip_unit_interval) {
  if (lags < 1) throw std::invalid_argument("ar_lags must be >= 1");
  const std::size_t n = target.size();
  if (fit_mask.size() != n) throw std::invalid_argument("fit_mask must have one entry per row");
  const std::size_t d = lags + 1;

  std::vector<double> x(n * d, 1.0);  // last column: intercept
  for (std::size_t k = 0; k < lags; ++k) {
    const auto col = lag(target, groups, horizon + static_cast<std::int64_t>(k));
    for (std::size_t i = 0; i < n; ++i) x[i * d + k] = col[i];
  }
  std::vector<std::uint8_t> valid_x(n, 1);
  std::vector<std::size_t> fit_rows;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < lags; ++k) {
      if (std::isnan(x[i * d + k])) {
        valid_x[i] = 0;
        break;
      }
    }
    if (fit_mask[i] != 0 && valid_x[i] != 0 && !std::isnan(target[i])) fit_rows.push_back(i);
  }

  std::vector<double> preds(n, kNaN);
  if (fit_rows.empty()) return preds;
  const std::vector<double> pooled = fit_ridge(x, target, d, fit_rows, alpha);

  auto predict = [&](std::size_t i, const std::vector<double>& w) {
    double s = 0.0;
    for (std::size_t k = 0; k < d; ++k) s += x[i * d + k] * w[k];
    if (clip_unit_interval) s = std::clamp(s, 0.0, 1.0);
    preds[i] = s;
  };

  if (!groups) {
    for (std::size_t i = 0; i < n; ++i) {
      if (valid_x[i] != 0) predict(i, pooled);
    }
    return preds;
  }
  const GroupRows g = group_rows(n, groups);
  const std::size_t min_rows = lags + 5;
  for_each_group(g, [&](std::size_t b, std::size_t e) {
    std::vector<std::size_t> g_fit;
    bool any_pred = false;
    for (std::size_t p = b; p < e; ++p) {
      const std::size_t i = g.rows[p];
      if (fit_mask[i] != 0 && valid_x[i] != 0 && !std::isnan(target[i])) g_fit.push_back(i);
      any_pred = any_pred || valid_x[i] != 0;
    }
    if (!any_pred) return;
    const std::vector<double> w = g_fit.size() >= min_rows ? fit_ridge(x, target, d, g_fit, alpha) : pooled;
    for (std::size_t p = b; p < e; ++p) {
      if (valid_x[g.rows[p]] != 0) predict(g.rows[p], w);
    }
  });
  return preds;
}

std::optional<std::int64_t> estimate_season(std::span<const double> target, GroupCodes groups,
                                            std::span<const std::uint8_t> fit_mask, std::int64_t max_lag,
                                            std::int64_t min_lag) {
  const std::size_t n = target.size();
  if (fit_mask.size() != n) throw std::invalid_argument("fit_mask must have one entry per row");
  if (min_lag < 0 || max_lag < 0) throw std::invalid_argument("lags must be non-negative");
  const GroupRows g = group_rows(n, groups);

  // Per group: fit rows with a value, in order, kept if long enough and not constant, then centered.
  std::vector<std::vector<double>> series;
  for (std::size_t k = 0; k < g.n_groups(); ++k) {
    std::vector<double> s;
    for (std::size_t p = g.offsets[k]; p < g.offsets[k + 1]; ++p) {
      const std::size_t i = g.rows[p];
      if (fit_mask[i] != 0 && !std::isnan(target[i])) s.push_back(target[i]);
    }
    if (s.size() <= static_cast<std::size_t>(4 * min_lag)) continue;
    const double mu = numeric::mean(s);
    if (!std::isfinite(mu)) continue;  // np.std is NaN with +-inf present, and NaN > 0 is false
    // np.std(s) > 0  <=>  some element differs from the (pairwise) mean.
    if (std::none_of(s.begin(), s.end(), [mu](double v) { return v != mu; })) continue;
    for (double& v : s) v -= mu;
    series.push_back(std::move(s));
  }
  if (series.empty()) return std::nullopt;

  std::size_t longest = 0;
  for (const auto& s : series) longest = std::max(longest, s.size());
  const std::size_t top = std::min(static_cast<std::size_t>(max_lag), longest / 2);
  std::vector<double> num(top + 1, 0.0);
  double den = 0.0;
  for (const auto& s : series) {
    const std::vector<double> acov = detail::circular_autocov(s, 2 * s.size());
    const std::size_t len = std::min(s.size(), top + 1);
    for (std::size_t i = 0; i < len; ++i) num[i] += acov[i];
    den += numeric::pairwise_sum(s.size(), [&](std::size_t i) { return s[i] * s[i]; });
  }
  std::vector<double> ac(top + 1);
  for (std::size_t i = 0; i <= top; ++i) ac[i] = num[i] / den;

  const auto lo = static_cast<std::size_t>(min_lag);
  if (lo > top) return std::nullopt;
  std::size_t start = top + 1;
  for (std::size_t i = lo; i <= top; ++i) {
    if (ac[i] < 0.5) {
      start = i;
      break;
    }
  }
  if (start > top) return std::nullopt;
  std::size_t peak = start;
  for (std::size_t i = start + 1; i <= top; ++i) {
    if (ac[i] > ac[peak]) peak = i;  // np.argmax: first maximum
  }
  double trough = ac[start];
  for (std::size_t i = start; i <= peak; ++i) trough = std::min(trough, ac[i]);
  if (ac[peak] < 0.3 || ac[peak] - trough < 0.2) return std::nullopt;
  return static_cast<std::int64_t>(peak);
}

}  // namespace paudit
