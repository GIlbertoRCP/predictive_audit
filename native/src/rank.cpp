#include "paudit/rank.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "confusion.hpp"
#include "paudit/numeric.hpp"

namespace paudit {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// (tps, fps) at every distinct score threshold, highest score first: what sklearn's
/// confusion_matrix_at_thresholds returns for the (weighted-by-repetition) data. Counts are
/// integers held exactly in doubles, so the order of rows within a tie never matters.
struct Curve {
  std::vector<double> tps;
  std::vector<double> fps;
};

Curve curve(std::span<const double> y, const ScoreOrder& so, std::span<const std::uint32_t> weights) {
  Curve c;
  c.tps.reserve(so.tie_ends.size());
  c.fps.reserve(so.tie_ends.size());
  std::uint64_t pos = 0;
  std::uint64_t neg = 0;
  std::size_t start = 0;
  for (const std::uint32_t end : so.tie_ends) {
    std::uint64_t run = 0;
    for (std::size_t k = start; k < end; ++k) {
      const std::uint32_t i = so.order[k];
      const std::uint64_t w = weights.empty() ? 1 : weights[i];
      run += w;
      if (y[i] == 1.0) {
        pos += w;
      } else {
        neg += w;
      }
    }
    if (run > 0) {  // a score that does not occur in the (resampled) data is not a threshold
      c.tps.push_back(static_cast<double>(pos));
      c.fps.push_back(static_cast<double>(neg));
    }
    start = end;
  }
  return c;
}

/// sklearn roc_curve(drop_intermediate=True) followed by auc(fpr, tpr).
double roc_auc_from(const Curve& c) {
  const std::size_t p = c.tps.size();
  if (p == 0 || c.tps.back() <= 0.0 || c.fps.back() <= 0.0) return kNaN;
  std::vector<double> tps{0.0};
  std::vector<double> fps{0.0};
  for (std::size_t j = 0; j < p; ++j) {
    bool keep = true;
    if (p > 2 && j > 0 && j + 1 < p) {
      const double d2f = (c.fps[j + 1] - c.fps[j]) - (c.fps[j] - c.fps[j - 1]);
      const double d2t = (c.tps[j + 1] - c.tps[j]) - (c.tps[j] - c.tps[j - 1]);
      keep = d2f != 0.0 || d2t != 0.0;
    }
    if (keep) {
      tps.push_back(c.tps[j]);
      fps.push_back(c.fps[j]);
    }
  }
  const double tp_total = tps.back();
  const double fp_total = fps.back();
  // np.trapezoid(tpr, fpr): sum of (dx * (y[i+1] + y[i]) / 2.0), pairwise.
  return numeric::pairwise_sum(tps.size() - 1, [&](std::size_t i) {
    const double dx = fps[i + 1] / fp_total - fps[i] / fp_total;
    return dx * (tps[i + 1] / tp_total + tps[i] / tp_total) / 2.0;
  });
}

/// sklearn precision_recall_curve followed by the step-function integral in
/// average_precision_score: max(0, -sum(diff(recall) * precision[:-1])).
double average_precision_from(const Curve& c) {
  const std::size_t p = c.tps.size();
  if (p == 0 || c.tps.back() <= 0.0 || c.fps.back() <= 0.0) return kNaN;
  const double tp_total = c.tps.back();
  // Reversed arrays with the (precision=1, recall=0) end point appended.
  auto precision_r = [&](std::size_t i) {
    if (i == p) return 1.0;
    const std::size_t j = p - 1 - i;
    const double ps = c.tps[j] + c.fps[j];
    return ps != 0.0 ? c.tps[j] / ps : 0.0;
  };
  auto recall_r = [&](std::size_t i) { return i == p ? 0.0 : c.tps[p - 1 - i] / tp_total; };
  const double s = numeric::pairwise_sum(
      p, [&](std::size_t i) { return (recall_r(i + 1) - recall_r(i)) * precision_r(i); });
  return std::max(0.0, -s);
}

bool is_thresholded(Metric m) noexcept {
  return m == Metric::f1 || m == Metric::mcc || m == Metric::balanced_accuracy || m == Metric::accuracy;
}

}  // namespace

ScoreOrder score_order(std::span<const double> scores) {
  const std::size_t n = scores.size();
  if (n > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("too many rows for rank metrics");
  ScoreOrder so;
  so.order.resize(n);
  std::iota(so.order.begin(), so.order.end(), 0U);
  std::stable_sort(so.order.begin(), so.order.end(),
                   [&](std::uint32_t a, std::uint32_t b) { return scores[a] > scores[b]; });
  for (std::size_t k = 1; k <= n; ++k) {
    if (k == n || scores[so.order[k]] != scores[so.order[k - 1]])
      so.tie_ends.push_back(static_cast<std::uint32_t>(k));
  }
  return so;
}

double roc_auc(std::span<const double> y_true, const ScoreOrder& order,
               std::span<const std::uint32_t> weights) {
  return roc_auc_from(curve(y_true, order, weights));
}

double average_precision(std::span<const double> y_true, const ScoreOrder& order,
                         std::span<const std::uint32_t> weights) {
  return average_precision_from(curve(y_true, order, weights));
}

void validate_rank_inputs(std::span<const double> y_true, std::span<const double> scores) {
  if (y_true.size() != scores.size())
    throw std::invalid_argument("y_true and y_score must have the same length");
  for (std::size_t i = 0; i < y_true.size(); ++i) {
    if (y_true[i] != 0.0 && y_true[i] != 1.0) throw std::invalid_argument("rank metrics need 0/1 labels");
    if (!std::isfinite(scores[i])) throw std::invalid_argument("rank metrics need finite scores");
  }
}

double roc_auc(std::span<const double> y_true, std::span<const double> scores) {
  validate_rank_inputs(y_true, scores);
  return roc_auc(y_true, score_order(scores));
}

double average_precision(std::span<const double> y_true, std::span<const double> scores) {
  validate_rank_inputs(y_true, scores);
  return average_precision(y_true, score_order(scores));
}

double optimal_threshold(Metric metric, std::span<const double> y_true, std::span<const double> y_score) {
  if (y_true.size() != y_score.size())
    throw std::invalid_argument("y_true and y_score must have the same length");
  if (!is_thresholded(metric)) return 0.5;
  std::vector<double> pos;
  std::vector<double> neg;
  std::vector<double> all;
  for (std::size_t i = 0; i < y_true.size(); ++i) {
    if (std::isnan(y_true[i]) || std::isnan(y_score[i])) continue;
    all.push_back(y_score[i]);
    (y_true[i] >= 0.5 ? pos : neg).push_back(y_score[i]);
  }
  const std::size_t n = all.size();
  if (n < 10) return 0.5;
  std::sort(all.begin(), all.end());
  if (all.back() - all.front() == 0.0) return 0.5;  // np.ptp(ys) == 0
  std::vector<double> uniq;
  for (const double v : all) {
    if (uniq.empty() || v != uniq.back()) uniq.push_back(v);
  }
  if (uniq.size() <= 2) return 0.5;

  std::vector<double> candidates;
  if (uniq.size() <= 100) {
    candidates = uniq;
  } else {
    // np.quantile(ys, np.linspace(0.02, 0.98, 97)) with linear interpolation.
    constexpr std::size_t kNum = 97;
    const double start = 0.02;
    const double stop = 0.98;
    const double step = (stop - start) / static_cast<double>(kNum - 1);
    for (std::size_t i = 0; i < kNum; ++i) {
      const double q = i + 1 == kNum ? stop : static_cast<double>(i) * step + start;
      const double vi = static_cast<double>(n - 1) * q;
      const double fl = std::floor(vi);
      if (vi >= static_cast<double>(n - 1)) {
        candidates.push_back(all[n - 1]);
        continue;
      }
      const auto lo = static_cast<std::size_t>(fl);
      candidates.push_back(numeric::lerp(all[lo], all[lo + 1], vi - fl));
    }
  }

  std::sort(pos.begin(), pos.end());
  std::sort(neg.begin(), neg.end());
  const auto n_pos = static_cast<double>(pos.size());
  const auto n_neg = static_cast<double>(neg.size());
  auto value_at = [&](double t) {
    // Predictions are positive where score >= t.
    const auto tp = static_cast<double>(pos.end() - std::lower_bound(pos.begin(), pos.end(), t));
    const auto fp = static_cast<double>(neg.end() - std::lower_bound(neg.begin(), neg.end(), t));
    const double fn = n_pos - tp;
    const double tn = n_neg - fp;
    switch (metric) {
      case Metric::f1:
        return detail::f1_from(tp, fp, fn);
      case Metric::mcc:
        return detail::mcc_from(tp, fp, fn, tn);
      case Metric::balanced_accuracy:
        return detail::balanced_accuracy_from(tp, fp, fn, tn);
      default:
        return (tp + tn) / static_cast<double>(n);
    }
  };
  double best_t = 0.5;
  double best_val = value_at(0.5);
  for (const double t : candidates) {
    const double v = value_at(t);
    if (std::isfinite(v) && v > best_val) {
      best_val = v;
      best_t = t;
    }
  }
  return best_t;
}

}  // namespace paudit
