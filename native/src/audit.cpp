#include "paudit/audit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "paudit/baselines.hpp"
#include "paudit/bootstrap.hpp"
#include "paudit/inference.hpp"
#include "paudit/numeric.hpp"
#include "paudit/parallel.hpp"
#include "paudit/rank.hpp"
#include "pyformat.hpp"

namespace paudit {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

using detail::py_fixed;
using detail::py_general;
using detail::py_percent0;

std::string metric_str(Metric m) { return std::string(metric_name(m)); }

/// Python's `_nan_low`: NaN sorts below everything.
double nan_low(double x) noexcept { return std::isnan(x) ? kNegInf : x; }

/// Ordered name -> values map with Python dict semantics (assigning an existing key keeps its slot).
struct Predictions {
  std::vector<std::string> names;
  std::vector<std::vector<double>> values;

  void set(const std::string& name, std::vector<double> v) {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it != names.end()) {
      values[static_cast<std::size_t>(it - names.begin())] = std::move(v);
      return;
    }
    names.push_back(name);
    values.push_back(std::move(v));
  }
  [[nodiscard]] const std::vector<double>& at(const std::string& name) const {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) throw std::logic_error("unknown prediction " + name);
    return values[static_cast<std::size_t>(it - names.begin())];
  }
};

// --- statistics used by the checks ---------------------------------------------------------------

/// np.std(x) > 0 for a NaN-free array: some element differs from its (pairwise) mean.
bool varies(std::span<const double> x) {
  const double mu = numeric::mean(x);
  if (!std::isfinite(mu)) return false;
  return std::any_of(x.begin(), x.end(), [mu](double v) { return v != mu; });
}

/// np.corrcoef(a, b)[0, 1] (sample covariance scaled by 1/(n-1), divided by both deviations, clipped).
double pearson(std::span<const double> a, std::span<const double> b) {
  const std::size_t n = a.size();
  const double ma = numeric::mean(a);
  const double mb = numeric::mean(b);
  std::array<double, 3> s{};
  auto term = [&](std::size_t i, std::array<double, 3>& v) {
    const double da = a[i] - ma;
    const double db = b[i] - mb;
    v = {da * da, db * db, da * db};
  };
  numeric::pairwise_sum_k<3>(term, 0, n, s);
  const double inv = 1.0 / static_cast<double>(n - 1);
  const double caa = s[0] * inv;
  const double cbb = s[1] * inv;
  const double cab = s[2] * inv;
  const double r = cab / std::sqrt(caa) / std::sqrt(cbb);
  return std::clamp(r, -1.0, 1.0);
}

/// scipy.stats.rankdata(method="average").
std::vector<double> average_ranks(std::span<const double> x) {
  // Tied values share one average rank, so the order of ties is irrelevant: sort (value, row)
  // pairs directly instead of an indirect stable sort.
  const std::size_t n = x.size();
  std::vector<std::pair<double, std::size_t>> sorted(n);
  for (std::size_t i = 0; i < n; ++i) sorted[i] = {x[i], i};
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<double> ranks(n);
  std::size_t s = 0;
  while (s < n) {
    std::size_t e = s + 1;
    while (e < n && sorted[e].first == sorted[s].first) ++e;
    const double r = 0.5 * (static_cast<double>(e) + static_cast<double>(s) + 1.0);
    for (std::size_t k = s; k < e; ++k) ranks[sorted[k].second] = r;
    s = e;
  }
  return ranks;
}

/// Rows where neither input is NaN.
std::pair<std::vector<double>, std::vector<double>> paired_finite(std::span<const double> a,
                                                                  std::span<const double> b) {
  std::vector<double> xa;
  std::vector<double> xb;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isnan(a[i]) && !std::isnan(b[i])) {
      xa.push_back(a[i]);
      xb.push_back(b[i]);
    }
  }
  return {std::move(xa), std::move(xb)};
}

/// predictive_audit.audit._corr
double corr(std::span<const double> a, std::span<const double> b) {
  auto [xa, xb] = paired_finite(a, b);
  if (xa.size() < 3 || !varies(xa) || !varies(xb)) return kNaN;
  return pearson(xa, xb);
}

/// predictive_audit.audit._spearman (scipy.stats.spearmanr: Pearson correlation of average ranks)
double spearman(std::span<const double> a, std::span<const double> b) {
  auto [xa, xb] = paired_finite(a, b);
  if (xa.size() < 5 || !varies(xa) || !varies(xb)) return kNaN;
  return pearson(average_ranks(xa), average_ranks(xb));
}

// --- checks ----------------------------------------------------------------------------------------

Check label_memory(Task task, Metric metric, std::span<const double> y, std::span<const double> persist,
                   std::int64_t horizon, double threshold) {
  const double autocorr = corr(y, persist);
  const double persist_value = compute(metric, y, persist, threshold);
  Check c{"label memory", Status::info, "", {}};
  c.details.emplace_back("lag_autocorrelation", autocorr);
  c.details.emplace_back("persistence_" + metric_str(metric), persist_value);
  if (task == Task::classification) {
    std::size_t same = 0;
    for (std::size_t i = 0; i < y.size(); ++i) same += y[i] == persist[i] ? 1 : 0;
    c.details.emplace_back("agreement_with_current_state",
                           static_cast<double>(same) / static_cast<double>(y.size()));
  }
  const bool bounded = !lower_is_better(metric);
  const bool high = (bounded && persist_value >= 0.9) || (std::isfinite(autocorr) && autocorr >= 0.95);
  const std::string h = std::to_string(horizon);
  if (high) {
    c.status = Status::warn;
    c.summary = "The target at t+" + h + " is mostly a copy of the target at t (lag-" + h +
                " autocorrelation " + py_fixed(autocorr, 3) + ", persistence " + metric_str(metric) + " " +
                py_fixed(persist_value, 3) +
                "). Any model will post a high score here; only the margin over persistence means anything.";
  } else {
    c.summary = "Lag-" + h + " autocorrelation of the target is " + py_fixed(autocorr, 3) +
                "; persistence scores " + metric_str(metric) + " " + py_fixed(persist_value, 3) + ".";
  }
  return c;
}

Check beats_baselines(const AuditConfig& cfg, Metric metric, std::span<const double> y,
                      std::span<const double> model, std::span<const double> baseline,
                      const std::string& baseline_name, double model_raw, double base_raw, GroupCodes groups,
                      const TimeKeys& times, double threshold) {
  Check c{"beats trivial baselines", Status::info, "", {}};
  const auto [mn, mx] = std::minmax_element(y.begin(), y.end());
  if (*mx - *mn == 0.0) {  // np.ptp(y_true) == 0
    c.summary = "The target is constant (" + py_general(y[0]) +
                ") over the scored rows, so there is nothing to forecast.";
    c.details.emplace_back("baseline", baseline_name);
    return c;
  }
  const bool lower = lower_is_better(metric);
  const double diff = (lower ? -model_raw : model_raw) - (lower ? -base_raw : base_raw);

  std::int64_t b_len = 0;
  if (cfg.block_mode == AuditConfig::BlockMode::politis_white) {
    std::vector<double> d_step;
    if (auto d = step_loss_differential(metric, y, model, baseline, threshold)) {
      d_step = std::move(*d);
    } else {
      d_step.resize(y.size());
      for (std::size_t i = 0; i < y.size(); ++i) d_step[i] = y[i] - baseline[i];
    }
    const PanelInfo panel = detect_aligned_panel(groups, times);
    if (panel.aligned) {
      std::vector<double> sums(panel.n_times, 0.0);
      std::vector<double> counts(panel.n_times, 0.0);
      for (std::size_t i = 0; i < d_step.size(); ++i) {
        const auto t = static_cast<std::size_t>(panel.time_index[i]);
        sums[t] += d_step[i];
        counts[t] += 1.0;
      }
      std::vector<double> series(panel.n_times);
      for (std::size_t t = 0; t < panel.n_times; ++t) series[t] = sums[t] / std::max(counts[t], 1.0);
      d_step = std::move(series);
    }
    b_len = static_cast<std::int64_t>(estimate_block_length(d_step));
  } else if (cfg.block_mode == AuditConfig::BlockMode::fixed) {
    b_len = cfg.block;
  } else {
    b_len = static_cast<std::int64_t>(default_block_length(y.size(), groups, times));
  }

  const Interval ci =
      block_bootstrap_diff(metric, y, model, baseline, b_len, cfg.n_boot, cfg.seed, groups, times, threshold);
  const DmResult dm = diebold_mariano(metric, y, model, baseline, cfg.horizon, groups, times, threshold);

  c.details.emplace_back("baseline", baseline_name);
  c.details.emplace_back("difference", diff);
  c.details.emplace_back("ci95", std::vector<double>{ci.lo, ci.hi});
  c.details.emplace_back("block_length", b_len);
  c.details.emplace_back("dm_stat", dm.stat);
  c.details.emplace_back("dm_p_value", dm.p_value);
  if (lower && base_raw > 0.0) c.details.emplace_back("skill_score", 1.0 - (model_raw / base_raw));

  const std::string m = metric_str(metric);
  const std::string what = lower ? m + " reduction vs " + baseline_name : m + " gain over " + baseline_name;
  const std::string dm_note = std::isfinite(dm.p_value) ? ", DM p=" + py_general(dm.p_value, 3) : "";
  const std::string margin = what + " = " + py_fixed(diff, 4, true) + " (95% block-bootstrap CI " +
                             py_fixed(ci.lo, 4, true) + " to " + py_fixed(ci.hi, 4, true) + dm_note + ")";
  if (!std::isfinite(diff) || diff <= 0) {
    c.status = Status::fail;
    c.summary = "Does not beat the best trivial baseline: " + margin + ".";
  } else if (!std::isfinite(ci.lo) || ci.lo <= 0) {
    c.status = Status::warn;
    c.summary = "Ahead of the best trivial baseline, but the margin is within noise: " + margin + ".";
  } else {
    c.status = Status::pass;
    c.summary = "Beats the best trivial baseline: " + margin + ".";
  }
  return c;
}

Check reported_score(Metric metric, double reported, const std::vector<std::string>& baseline_names,
                     const std::vector<double>& baseline_values) {
  const bool lower = lower_is_better(metric);
  std::size_t best = 0;
  double best_key = nan_low(lower ? -baseline_values[0] : baseline_values[0]);
  for (std::size_t k = 1; k < baseline_names.size(); ++k) {
    const double key = nan_low(lower ? -baseline_values[k] : baseline_values[k]);
    if (key > best_key) {
      best_key = key;
      best = k;
    }
  }
  const std::string& name = baseline_names[best];
  const double value = baseline_values[best];
  const double gap = lower ? value - reported : reported - value;
  Check c{"reported score", Status::info, "", {}};
  c.details.emplace_back("reported", reported);
  c.details.emplace_back("baseline", name);
  c.details.emplace_back("baseline_value", value);
  c.details.emplace_back("gap", gap);
  const std::string m = metric_str(metric);
  const std::string text = "Reported " + m + " " + py_general(reported, 4) + "; " + name + " alone gets " +
                           py_general(value, 4) + " on the same rows";
  const double tolerance = lower ? 0.05 * std::abs(value) : 0.02;
  if (!std::isfinite(value)) {
    c.summary = m + " is undefined on these rows.";
  } else if (gap <= 0) {
    c.status = Status::fail;
    c.summary = text + ". The claim is not evidence of forecasting skill.";
  } else if (gap < tolerance) {
    c.status = Status::warn;
    c.summary = text + ". The claimed improvement is marginal.";
  } else {
    c.status = Status::pass;
    c.summary = text + ".";
  }
  return c;
}

/// Rows i and i - 1 have equal time keys.
bool same_time(const KeyColumn& t, const std::vector<std::size_t>& perm, std::size_t i) {
  return std::visit([&](const auto& col) { return col[perm[i]] == col[perm[i - 1]]; }, t);
}

Check temporal_split(const std::vector<std::int64_t>* gkey, std::int64_t horizon,
                     const std::optional<KeyColumn>& time, const std::vector<std::size_t>& perm,
                     const std::vector<std::uint8_t>& train, const std::vector<std::uint8_t>& test) {
  Check c{"temporal split", Status::info, "", {}};
  const std::size_t n = train.size();
  if (std::none_of(train.begin(), train.end(), [](std::uint8_t v) { return v != 0; })) {
    c.summary = "No rows are labelled 'train', so there is no overlap to check.";
    return c;
  }
  std::vector<std::int64_t> g(n, 0);
  std::vector<std::uint8_t> grp_start(n, 0);
  std::vector<std::int64_t> clock(n);
  std::size_t n_groups = 1;
  if (gkey == nullptr) {
    for (std::size_t i = 0; i < n; ++i) clock[i] = static_cast<std::int64_t>(i);
    grp_start[0] = 1;
  } else {
    g = *gkey;
    n_groups = static_cast<std::size_t>(*std::max_element(g.begin(), g.end()) + 1);
    std::int64_t run_start = 0;
    for (std::size_t i = 0; i < n; ++i) {
      grp_start[i] = (i == 0 || g[i] != g[i - 1]) ? 1 : 0;
      if (grp_start[i] != 0) run_start = static_cast<std::int64_t>(i);
      clock[i] = static_cast<std::int64_t>(i) - run_start;
    }
  }

  std::vector<std::int64_t> last_train(n_groups, -1);
  for (std::size_t i = 0; i < n; ++i) {
    if (train[i] != 0) {
      auto& lt = last_train[static_cast<std::size_t>(g[i])];
      lt = std::max(lt, clock[i]);
    }
  }
  std::size_t n_test = 0;
  std::size_t n_before = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (test[i] == 0) continue;
    ++n_test;
    const auto lt = last_train[static_cast<std::size_t>(g[i])];
    n_before += (lt >= 0 && clock[i] < lt) ? 1 : 0;
  }
  const double share_before = static_cast<double>(n_before) / static_cast<double>(n_test);

  std::vector<double> runs(n_groups, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const bool prev_test = i > 0 && test[i - 1] != 0;
    if (test[i] != 0 && (!prev_test || grp_start[i] != 0)) runs[static_cast<std::size_t>(g[i])] += 1.0;
  }
  std::vector<double> active;
  for (const double r : runs) {
    if (r > 0) active.push_back(r);
  }
  const double mean_runs = active.empty() ? 0.0 : numeric::mean(active);

  std::int64_t embargo_violated = 0;
  if (horizon > 1) {
    const auto none = static_cast<std::int64_t>(n) + 1;
    std::vector<std::int64_t> first_test(n_groups, none);
    for (std::size_t i = 0; i < n; ++i) {
      if (test[i] != 0) {
        auto& ft = first_test[static_cast<std::size_t>(g[i])];
        ft = std::min(ft, clock[i]);
      }
    }
    for (std::size_t k = 0; k < n_groups; ++k) {
      if (last_train[k] >= 0 && first_test[k] <= static_cast<std::int64_t>(n) &&
          first_test[k] - last_train[k] < horizon) {
        ++embargo_violated;
      }
    }
  }

  std::int64_t dup_timestamps = 0;
  if (time && n > 1) {
    for (std::size_t i = 1; i < n; ++i) {
      if (grp_start[i] == 0 && same_time(*time, perm, i)) ++dup_timestamps;
    }
  }

  c.details.emplace_back("test_rows_before_last_train_row", share_before);
  c.details.emplace_back("mean_test_segments_per_group", mean_runs);
  c.details.emplace_back("embargo_violations", embargo_violated);
  c.details.emplace_back("duplicate_timestamps", dup_timestamps);

  if (share_before == 0) {
    if (embargo_violated > 0) {
      c.status = Status::warn;
      c.summary = "Test rows follow training rows, but " + std::to_string(embargo_violated) +
                  " series have a train/test gap smaller than horizon=" + std::to_string(horizon) +
                  ", so late training targets overlap the test period.";
    } else if (dup_timestamps > 0) {
      c.status = Status::warn;
      c.summary = "Every test row follows training rows, but " + std::to_string(dup_timestamps) +
                  " duplicate timestamps were found within series (lag alignment may be ambiguous).";
    } else {
      c.status = Status::pass;
      c.summary = "Every test row comes after the training data in time.";
    }
  } else if (mean_runs <= 3) {
    c.status = Status::warn;
    c.summary =
        py_percent0(share_before) +
        " of test rows precede training rows, in a few contiguous blocks. Fine for blocked cross-validation "
        "with a gap; otherwise information leaks backward in time.";
  } else {
    c.status = Status::fail;
    c.summary = "Test rows are interleaved with training rows (" + py_percent0(share_before) +
                " precede the last training row, ~" + py_fixed(mean_runs, 0) +
                " separate test segments per series). This is a shuffled split: with overlapping windows or "
                "autocorrelated rows, test examples are near-copies of training examples.";
  }
  return c;
}

Check feature_leakage(const std::vector<std::pair<std::string, std::vector<double>>>& features,
                      std::span<const double> y, GroupCodes groups, std::int64_t horizon) {
  const std::vector<double> y_obs = lag(y, groups, horizon);
  std::vector<std::vector<double>> future(static_cast<std::size_t>(std::max<std::int64_t>(horizon, 0)));
  for (std::size_t k = 0; k < future.size(); ++k) future[k] = lag(y, groups, static_cast<std::int64_t>(k));

  const std::size_t nf = features.size();
  const std::size_t nk = future.size();
  // Every correlation the reference might look at, computed in parallel: per feature, Pearson and
  // Spearman against the observed target (slot 0) and each unobserved future target (slots 1..nk),
  // plus the feature's count of distinct values. The leak decisions below then follow the
  // reference's order exactly; computing a Spearman it would skip changes nothing.
  const std::size_t per_feature = 2 * (nk + 1) + 1;
  std::vector<double> stats(nf * per_feature, kNaN);
  parallel::parallel_for(nf * per_feature, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t task = b; task < e; ++task) {
      const std::size_t fi = task / per_feature;
      const std::size_t slot = task % per_feature;
      const std::vector<double>& x = features[fi].second;
      if (slot == per_feature - 1) {
        std::vector<double> present;
        for (const double v : x) {
          if (!std::isnan(v)) present.push_back(v);
        }
        std::sort(present.begin(), present.end());
        stats[task] = static_cast<double>(std::unique(present.begin(), present.end()) - present.begin());
        continue;
      }
      const std::size_t which = slot / 2;  // 0 = observed, k + 1 = future[k]
      const std::vector<double>& target = which == 0 ? y_obs : future[which - 1];
      stats[task] = std::abs(slot % 2 == 0 ? corr(x, target) : spearman(x, target));
    }
  });

  std::vector<std::vector<std::string>> leaks(nf);
  std::vector<double> strongest(nf, 0.0);
  for (std::size_t fi = 0; fi < nf; ++fi) {
    const double* st = stats.data() + fi * per_feature;
    const std::string& name = features[fi].first;
    double best = 0.0;
    const double obs_r = st[0];
    const double obs_rho = st[1];
    const auto n_unique = static_cast<std::size_t>(st[per_feature - 1]);
    for (std::size_t k = 0; k < nk; ++k) {
      const std::string tag = name + " ~ y[t+" + std::to_string(horizon - static_cast<std::int64_t>(k)) + "]";
      const double r = st[2 * (k + 1)];
      if (std::isfinite(r) && r > best) best = r;
      if (std::isfinite(r) && r >= 0.999 && (!std::isfinite(obs_r) || r > obs_r + 1e-4 || obs_r < 0.999)) {
        leaks[fi].push_back(tag + " (|r|=" + py_fixed(r, 4) + ")");
        continue;
      }
      if (n_unique >= 10) {
        const double rho = st[2 * (k + 1) + 1];
        if (std::isfinite(rho) && rho > best) best = rho;
        if (std::isfinite(rho) && rho >= 0.999 && (!std::isfinite(obs_rho) || rho > obs_rho + 1e-4)) {
          leaks[fi].push_back(tag + " (monotonic |rho|=" + py_fixed(rho, 4) + ")");
        }
      }
    }
    strongest[fi] = best;
  }

  Check c{"feature leakage", Status::pass, "", {}};
  NamedDoubles strongest_by_name;
  for (std::size_t fi = 0; fi < nf; ++fi) {
    // Python dict semantics: a repeated feature name keeps its first slot, last value.
    const auto it = std::find_if(strongest_by_name.begin(), strongest_by_name.end(),
                                 [&](const auto& kv) { return kv.first == features[fi].first; });
    if (it != strongest_by_name.end()) {
      it->second = strongest[fi];
    } else {
      strongest_by_name.emplace_back(features[fi].first, strongest[fi]);
    }
  }
  c.details.emplace_back("max_abs_corr_with_future_target", std::move(strongest_by_name));
  std::string joined;
  for (const auto& per : leaks) {
    for (const auto& l : per) joined += (joined.empty() ? "" : "; ") + l;
  }
  if (!joined.empty()) {
    c.status = Status::fail;
    c.summary = "Features copy future target values: " + joined + ".";
  } else {
    c.summary = "No feature is a near-copy of a future target value.";
  }
  return c;
}

std::optional<Check> scaler_leakage(const std::vector<std::pair<std::string, std::vector<double>>>& cols,
                                    const std::vector<std::uint8_t>& train,
                                    const std::vector<std::uint8_t>& test) {
  const bool any_train = std::any_of(train.begin(), train.end(), [](std::uint8_t v) { return v != 0; });
  const bool any_test = std::any_of(test.begin(), test.end(), [](std::uint8_t v) { return v != 0; });
  if (!any_train || !any_test) return std::nullopt;
  std::vector<std::string> suspect;
  for (const auto& [name, vals] : cols) {
    std::vector<double> tr;
    std::vector<double> te;
    for (std::size_t i = 0; i < vals.size(); ++i) {
      if (std::isnan(vals[i])) continue;
      if (train[i] != 0) tr.push_back(vals[i]);
      if (test[i] != 0) te.push_back(vals[i]);
    }
    if (tr.size() < 10 || te.size() < 5) continue;
    const auto [tr_lo, tr_hi] = std::minmax_element(tr.begin(), tr.end());
    const auto [te_lo, te_hi] = std::minmax_element(te.begin(), te.end());
    const double tr_min = *tr_lo;
    const double tr_max = *tr_hi;
    const double te_min = *te_lo;
    const double te_max = *te_hi;
    const double gmin = std::min(tr_min, te_min);
    const double gmax = std::max(tr_max, te_max);
    if (!(gmin >= -1e-6 && gmax <= 1.0 + 1e-6 && gmin <= 1e-6 && gmax >= 1.0 - 1e-6)) continue;
    auto interior = [](double v) { return v > 1e-6 && v < 1.0 - 1e-6; };
    if (!(std::any_of(tr.begin(), tr.end(), interior) || std::any_of(te.begin(), te.end(), interior)))
      continue;
    const bool test_sets_min = te_min <= 1e-6 && tr_min > 1e-3;
    const bool test_sets_max = te_max >= 1.0 - 1e-6 && tr_max < 1.0 - 1e-3;
    if (test_sets_min || test_sets_max) suspect.push_back(name);
  }
  if (suspect.empty()) return std::nullopt;
  std::string joined;
  for (const auto& s : suspect) joined += (joined.empty() ? "" : ", ") + s;
  Check c{
      "scaler leakage",
      Status::warn,
      "Columns appear min-max scaled across train+test combined (exact 0/1 boundary occurs only in test): " +
          joined + ".",
      {}};
  c.details.emplace_back("suspect_columns", std::move(suspect));
  return c;
}

// --- pipeline helpers -------------------------------------------------------------------------------

std::size_t key_size(const KeyColumn& k) {
  return std::visit([](const auto& c) { return c.size(); }, k);
}

int compare_key(const KeyColumn& k, std::size_t a, std::size_t b) {
  return std::visit(
      [&](const auto& c) {
        if (c[a] < c[b]) return -1;
        if (c[b] < c[a]) return 1;
        return 0;
      },
      k);
}

/// Stable order of rows by (group keys..., time), like df.sort_values(cols, kind="stable").
std::vector<std::size_t> sort_order(const AuditColumns& c, std::size_t n) {
  std::vector<const KeyColumn*> keys;
  for (const auto& k : c.group_keys) keys.push_back(&k);
  if (c.time) keys.push_back(&*c.time);
  std::vector<std::size_t> perm(n);
  std::iota(perm.begin(), perm.end(), std::size_t{0});
  if (keys.empty()) return perm;
  auto less = [&](std::size_t a, std::size_t b) {
    for (const KeyColumn* k : keys) {
      const int cmp = compare_key(*k, a, b);
      if (cmp != 0) return cmp < 0;
    }
    return false;
  };
  bool sorted = true;
  for (std::size_t i = 1; i < n && sorted; ++i) sorted = !less(i, i - 1);
  if (!sorted) std::stable_sort(perm.begin(), perm.end(), less);
  return perm;
}

std::vector<double> take(std::span<const double> v, const std::vector<std::size_t>& perm) {
  std::vector<double> out(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i) out[i] = v[perm[i]];
  return out;
}

std::vector<double> subset(const std::vector<double>& v, const std::vector<std::size_t>& rows) {
  std::vector<double> out(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) out[i] = v[rows[i]];
  return out;
}

}  // namespace

std::string_view status_name(Status s) noexcept {
  switch (s) {
    case Status::info:
      return "INFO";
    case Status::pass:
      return "PASS";
    case Status::warn:
      return "WARN";
    case Status::fail:
      return "FAIL";
  }
  return "INFO";
}

Status AuditReport::verdict() const noexcept {
  Status v = Status::info;
  bool first = true;
  for (const auto& c : checks) {
    if (first || c.status > v) v = c.status;
    first = false;
  }
  return v;
}

AuditReport audit(const AuditColumns& c, const AuditConfig& cfg) {
  const std::size_t n = c.target.size();
  auto check_len = [n](std::size_t len, const char* what) {
    if (len != n) throw std::invalid_argument(std::string(what) + " must have one entry per row");
  };
  if (c.pred) check_len(c.pred->size(), "pred");
  for (const auto& k : c.group_keys) check_len(key_size(k), "group keys");
  if (c.time) check_len(key_size(*c.time), "time");
  if (c.split) check_len(c.split->size(), "split");
  for (const auto& f : c.features) check_len(f.values.size(), "features");
  for (const auto& b : c.extra_baselines) check_len(b.values.size(), "extra baselines");

  const std::vector<std::size_t> perm = sort_order(c, n);

  std::optional<std::vector<std::int64_t>> gkey;
  if (!c.group_keys.empty()) {
    gkey.emplace(n, 0);
    std::int64_t code = 0;
    for (std::size_t i = 1; i < n; ++i) {
      for (const auto& k : c.group_keys) {
        if (compare_key(k, perm[i], perm[i - 1]) != 0) {
          ++code;
          break;
        }
      }
      (*gkey)[i] = code;
    }
  }
  const GroupCodes groups = gkey ? GroupCodes(std::span<const std::int64_t>(*gkey)) : std::nullopt;

  const std::vector<double> y = take(c.target, perm);
  if (cfg.task == Task::classification) {
    for (const double v : y) {
      if (!std::isnan(v) && v != 0.0 && v != 1.0)
        throw std::invalid_argument("classification target must be binary 0/1");
    }
  }

  std::vector<std::uint8_t> test(n, 1);
  std::vector<std::uint8_t> train(n, 0);
  std::vector<std::uint8_t> fit(n, 1);
  if (c.split) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint8_t kind = (*c.split)[perm[i]];
      test[i] = kind == kSplitTest ? 1 : 0;
      train[i] = kind == kSplitTrain ? 1 : 0;
      fit[i] = test[i] != 0 ? 0 : 1;
    }
    if (std::none_of(test.begin(), test.end(), [](std::uint8_t v) { return v != 0; })) {
      throw std::invalid_argument("no rows with " + cfg.split_name + " == 'test'");
    }
  }
  if (cfg.season_error) throw std::invalid_argument(*cfg.season_error);

  std::optional<std::int64_t> season;
  if (cfg.season_mode == AuditConfig::SeasonMode::automatic) {
    season = estimate_season(y, groups, fit);
  } else if (cfg.season_mode == AuditConfig::SeasonMode::fixed) {
    season = cfg.season;
  }

  // --- baselines, in predictive_audit.baselines.all_baselines order -----------------------------------
  const std::int64_t h = cfg.horizon;
  const std::int64_t window = (cfg.window && *cfg.window != 0) ? *cfg.window : std::max<std::int64_t>(3, h);
  const bool clf = cfg.task == Task::classification;
  Predictions base;
  const std::vector<double> observed = lag(y, groups, h);
  base.set("persistence", observed);
  base.set(
      "moving average (" + std::to_string(window) + ")",
      rolling_mean(observed, groups, static_cast<std::size_t>(window), static_cast<std::size_t>(window)));
  {
    std::vector<double> fit_vals;
    for (std::size_t i = 0; i < n; ++i) {
      if (fit[i] != 0 && !std::isnan(y[i])) fit_vals.push_back(y[i]);
    }
    const double mu = numeric::mean(fit_vals);
    const double value = clf ? (mu >= 0.5 ? 1.0 : 0.0) : mu;
    base.set(clf ? "majority class" : "training mean", std::vector<double>(n, value));
  }
  if (cfg.include_ewma) {
    base.set("ewma (" + std::to_string(window) + ")",
             ewm_mean(observed, groups, static_cast<double>(window), cfg.ewm_com1_update));
  }
  if (season && *season != 0) {
    const std::int64_t m = *season;
    const std::int64_t k = (h + m - 1) / m * m;  // math.ceil(h / m) * m for m > 0
    base.set("seasonal naive (m=" + std::to_string(m) + ")", lag(y, groups, k));
  }
  if (cfg.ar_lags && *cfg.ar_lags != 0) {
    if (*cfg.ar_lags < 1) throw std::invalid_argument("ar_lags must be >= 1");
    base.set("linear AR(" + std::to_string(*cfg.ar_lags) + ")",
             linear_ar(y, groups, h, fit, static_cast<std::size_t>(*cfg.ar_lags), 1.0, clf));
  }
  for (const auto& extra : c.extra_baselines) base.set(extra.name, take(extra.values, perm));

  Predictions preds;
  std::optional<std::string> model_name;
  if (c.pred) {
    model_name = cfg.model_name;
    preds.set(*model_name, take(*c.pred, perm));
  }
  for (std::size_t k = 0; k < base.names.size(); ++k) preds.set(base.names[k], base.values[k]);

  double threshold = 0.5;
  if (clf) {
    if (cfg.threshold_auto) {
      const bool any_fit = std::any_of(fit.begin(), fit.end(), [](std::uint8_t v) { return v != 0; });
      if (model_name && any_fit) {
        std::vector<double> yf;
        std::vector<double> pf;
        const auto& pm = preds.at(*model_name);
        for (std::size_t i = 0; i < n; ++i) {
          if (fit[i] != 0) {
            yf.push_back(y[i]);
            pf.push_back(pm[i]);
          }
        }
        threshold = optimal_threshold(cfg.metric, yf, pf);
      }
    } else {
      threshold = cfg.threshold;
    }
  }

  std::vector<std::size_t> eval_rows;
  for (std::size_t i = 0; i < n; ++i) {
    bool ok = test[i] != 0 && !std::isnan(y[i]);
    for (std::size_t k = 0; ok && k < preds.values.size(); ++k) ok = !std::isnan(preds.values[k][i]);
    if (ok) eval_rows.push_back(i);
  }
  if (eval_rows.size() < 10) {
    throw std::invalid_argument("only " + std::to_string(eval_rows.size()) +
                                " rows can be scored; need test rows with enough history");
  }

  const std::vector<double> y_eval = subset(y, eval_rows);
  std::vector<std::vector<double>> p_eval(preds.values.size());
  for (std::size_t k = 0; k < preds.values.size(); ++k) p_eval[k] = subset(preds.values[k], eval_rows);
  auto p_of = [&](const std::string& name) -> const std::vector<double>& {
    const auto it = std::find(preds.names.begin(), preds.names.end(), name);
    return p_eval[static_cast<std::size_t>(it - preds.names.begin())];
  };
  std::optional<std::vector<std::int64_t>> g_eval;
  if (gkey) {
    g_eval.emplace(eval_rows.size());
    for (std::size_t i = 0; i < eval_rows.size(); ++i) (*g_eval)[i] = (*gkey)[eval_rows[i]];
  }
  std::vector<std::int64_t> t_eval_i;
  std::vector<double> t_eval_f;
  TimeKeys t_eval;
  if (c.time) {
    if (const auto* ti = std::get_if<std::span<const std::int64_t>>(&*c.time)) {
      for (const std::size_t r : eval_rows) t_eval_i.push_back((*ti)[perm[r]]);
      t_eval = std::span<const std::int64_t>(t_eval_i);
    } else {
      const auto& tf = std::get<std::span<const double>>(*c.time);
      for (const std::size_t r : eval_rows) t_eval_f.push_back(tf[perm[r]]);
      t_eval = std::span<const double>(t_eval_f);
    }
  }
  const GroupCodes g_eval_codes = g_eval ? GroupCodes(std::span<const std::int64_t>(*g_eval)) : std::nullopt;

  // --- metric table --------------------------------------------------------------------------------
  const std::vector<Metric> metric_list =
      clf ? std::vector<Metric>{Metric::f1,       Metric::pr_auc,  Metric::mcc,  Metric::balanced_accuracy,
                                Metric::accuracy, Metric::roc_auc, Metric::brier}
          : std::vector<Metric>{Metric::mae, Metric::rmse, Metric::r2, Metric::smape};
  AuditReport report;
  report.table.resize(preds.names.size());
  parallel::parallel_for(preds.names.size(), 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      report.table[k].first = preds.names[k];
      for (const Metric m : metric_list)
        report.table[k].second.emplace_back(m, compute(m, y_eval, p_eval[k], threshold));
    }
  });
  auto table_value = [&](const std::string& name, Metric m) {
    const auto it = std::find_if(report.table.begin(), report.table.end(),
                                 [&](const auto& row) { return row.first == name; });
    for (const auto& [mm, v] : it->second) {
      if (mm == m) return v;
    }
    return kNaN;
  };

  const bool lower = lower_is_better(cfg.metric);
  std::size_t best_idx = 0;
  double best_key = kNegInf;
  for (std::size_t k = 0; k < base.names.size(); ++k) {
    const double v = table_value(base.names[k], cfg.metric);
    const double key = nan_low(lower ? -v : v);
    if (k == 0 || key > best_key) {
      best_key = key;
      best_idx = k;
    }
  }
  const std::string best = base.names[best_idx];

  report.task = cfg.task;
  report.target = cfg.target_name;
  report.horizon = h;
  report.metric = cfg.metric;
  report.model = model_name;
  report.season = season;
  report.n_eval = eval_rows.size();
  report.best_baseline = best;
  if (clf) {
    report.prevalence = numeric::mean(y_eval);
    report.threshold = threshold;
  }
  if (!clf && model_name) {
    const double persist_mae = table_value("persistence", Metric::mae);
    const double model_mae = table_value(*model_name, Metric::mae);
    report.mase = persist_mae > 0.0 ? model_mae / persist_mae : kNaN;
  }

  // --- checks ----------------------------------------------------------------------------------------
  report.checks.push_back(label_memory(cfg.task, cfg.metric, y_eval, p_of("persistence"), h, threshold));
  if (model_name) {
    report.checks.push_back(beats_baselines(cfg, cfg.metric, y_eval, p_of(*model_name), p_of(best), best,
                                            table_value(*model_name, cfg.metric),
                                            table_value(best, cfg.metric), g_eval_codes, t_eval, threshold));
  }
  if (cfg.reported) {
    std::vector<double> values;
    for (const auto& name : base.names) values.push_back(table_value(name, cfg.metric));
    report.checks.push_back(reported_score(cfg.metric, *cfg.reported, base.names, values));
  }
  std::vector<std::pair<std::string, std::vector<double>>> features;
  for (const auto& f : c.features) features.emplace_back(f.name, take(f.values, perm));
  if (c.split) {
    report.checks.push_back(temporal_split(gkey ? &*gkey : nullptr, h, c.time, perm, train, test));
    std::vector<std::pair<std::string, std::vector<double>>> scaler_cols;
    scaler_cols.emplace_back(cfg.target_name, y);
    for (const auto& f : features) scaler_cols.push_back(f);
    if (auto s = scaler_leakage(scaler_cols, train, test)) report.checks.push_back(std::move(*s));
  }
  if (!features.empty()) report.checks.push_back(feature_leakage(features, y, groups, h));

  if (cfg.want_eval_arrays) {
    for (const std::size_t r : eval_rows) report.eval_rows.push_back(static_cast<std::int64_t>(perm[r]));
    report.eval_predictions = std::move(p_eval);
    report.eval_target = y_eval;
  }
  return report;
}

std::vector<AuditReport> audit_each_series(const AuditColumns& c, const AuditConfig& cfg,
                                           std::span<const std::int64_t> series, std::size_t n_series) {
  const std::size_t n = c.target.size();
  if (series.size() != n) throw std::invalid_argument("series must have one entry per row");
  std::vector<std::vector<std::size_t>> rows(n_series);
  for (std::size_t i = 0; i < n; ++i) {
    if (series[i] >= 0) rows[static_cast<std::size_t>(series[i])].push_back(i);
  }
  std::vector<AuditReport> reports(n_series);
  std::vector<std::exception_ptr> errors(n_series);
  parallel::parallel_for(n_series, 1, [&](std::size_t b, std::size_t e) {
    for (std::size_t s = b; s < e; ++s) {
      try {
        const auto& r = rows[s];
        // Owned per-series copies of every column, then spans over them.
        std::vector<double> target(r.size());
        for (std::size_t i = 0; i < r.size(); ++i) target[i] = c.target[r[i]];
        auto gather = [&](std::span<const double> v) {
          std::vector<double> out(r.size());
          for (std::size_t i = 0; i < r.size(); ++i) out[i] = v[r[i]];
          return out;
        };
        std::optional<std::vector<double>> pred;
        if (c.pred) pred = gather(*c.pred);
        std::optional<std::vector<std::int64_t>> time_i;
        std::optional<std::vector<double>> time_f;
        if (c.time) {
          if (const auto* ti = std::get_if<std::span<const std::int64_t>>(&*c.time)) {
            time_i.emplace(r.size());
            for (std::size_t i = 0; i < r.size(); ++i) (*time_i)[i] = (*ti)[r[i]];
          } else {
            time_f = gather(std::get<std::span<const double>>(*c.time));
          }
        }
        std::optional<std::vector<std::uint8_t>> split;
        if (c.split) {
          split.emplace(r.size());
          for (std::size_t i = 0; i < r.size(); ++i) (*split)[i] = (*c.split)[r[i]];
        }
        std::vector<std::vector<double>> feats;
        for (const auto& f : c.features) feats.push_back(gather(f.values));
        std::vector<std::vector<double>> extras;
        for (const auto& x : c.extra_baselines) extras.push_back(gather(x.values));

        AuditColumns sub;
        sub.target = target;
        if (pred) sub.pred = std::span<const double>(*pred);
        if (time_i) sub.time = KeyColumn(std::span<const std::int64_t>(*time_i));
        if (time_f) sub.time = KeyColumn(std::span<const double>(*time_f));
        if (split) sub.split = std::span<const std::uint8_t>(*split);
        for (std::size_t k = 0; k < feats.size(); ++k) sub.features.push_back({c.features[k].name, feats[k]});
        for (std::size_t k = 0; k < extras.size(); ++k)
          sub.extra_baselines.push_back({c.extra_baselines[k].name, extras[k]});
        reports[s] = audit(sub, cfg);
      } catch (...) {
        errors[s] = std::current_exception();
      }
    }
  });
  for (const auto& err : errors) {
    if (err) std::rethrow_exception(err);
  }
  return reports;
}

}  // namespace paudit
