#pragma once

// Floating-point building blocks that reproduce NumPy's evaluation order, so native
// results match the Python reference bit for bit wherever NumPy's order is defined.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace paudit::numeric {

/// NumPy's pairwise-summation block size (PW_BLOCKSIZE in loops_utils.h).
inline constexpr std::size_t kPairwiseBlock = 128;

/// Sum K quantities at once with NumPy's pairwise algorithm (`np.sum` / `np.add.reduce` over a
/// contiguous axis). `f(i, v)` writes the K terms of element i into `v`. Each of the K sums uses
/// exactly the same association order NumPy would use for that column alone.
template <std::size_t K, class F>
void pairwise_sum_k(F& f, std::size_t off, std::size_t n, std::array<double, K>& res) {
  std::array<double, K> v{};
  if (n < 8) {
    res.fill(0.0);
    for (std::size_t i = 0; i < n; ++i) {
      f(off + i, v);
      for (std::size_t k = 0; k < K; ++k) res[k] += v[k];
    }
    return;
  }
  if (n <= kPairwiseBlock) {
    std::array<std::array<double, K>, 8> r{};
    for (std::size_t j = 0; j < 8; ++j) f(off + j, r[j]);
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (std::size_t j = 0; j < 8; ++j) {
        f(off + i + j, v);
        for (std::size_t k = 0; k < K; ++k) r[j][k] += v[k];
      }
    }
    for (std::size_t k = 0; k < K; ++k) {
      res[k] = ((r[0][k] + r[1][k]) + (r[2][k] + r[3][k])) + ((r[4][k] + r[5][k]) + (r[6][k] + r[7][k]));
    }
    for (; i < n; ++i) {
      f(off + i, v);
      for (std::size_t k = 0; k < K; ++k) res[k] += v[k];
    }
    return;
  }
  std::size_t n2 = n / 2;
  n2 -= n2 % 8;
  std::array<double, K> a{};
  std::array<double, K> b{};
  pairwise_sum_k<K>(f, off, n2, a);
  pairwise_sum_k<K>(f, off + n2, n - n2, b);
  for (std::size_t k = 0; k < K; ++k) res[k] = a[k] + b[k];
}

/// Pairwise sum of `g(i)` for i in [0, n), matching `np.sum` on a contiguous float64 array.
template <class G>
[[nodiscard]] double pairwise_sum(std::size_t n, G&& g) {
  auto f = [&g](std::size_t i, std::array<double, 1>& v) { v[0] = g(i); };
  std::array<double, 1> res{};
  pairwise_sum_k<1>(f, 0, n, res);
  return res[0];
}

[[nodiscard]] inline double sum(std::span<const double> x) {
  return pairwise_sum(x.size(), [x](std::size_t i) { return x[i]; });
}

/// `np.mean` of a contiguous float64 array (NaN for empty input, as NumPy returns).
[[nodiscard]] inline double mean(std::span<const double> x) { return sum(x) / static_cast<double>(x.size()); }

/// Exclusive prefix sums with a leading zero: out[0] = 0, out[k] = x[0] + ... + x[k-1].
/// Sequential accumulation, matching `np.cumsum`.
inline void prefix_sums(std::span<const double> x, std::span<double> out) {
  if (out.size() != x.size() + 1) throw std::invalid_argument("prefix_sums: output must be len(x) + 1");
  out[0] = 0.0;
  double acc = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    acc += x[i];
    out[i + 1] = acc;
  }
}

/// NumPy's `_lerp`: a + (b - a) * t, re-anchored on b when t >= 0.5 for monotonicity.
[[nodiscard]] inline double lerp(double a, double b, double t) {
  const double diff = b - a;
  if (t >= 0.5) return b - diff * (1.0 - t);
  return a + diff * t;
}

/// `np.percentile(values, q)` with the default "linear" method. `values` must be NaN-free and
/// non-empty; it is reordered in place.
[[nodiscard]] inline double percentile_linear(std::vector<double>& values, double q_percent) {
  const std::size_t n = values.size();
  if (n == 0) throw std::invalid_argument("percentile of an empty array");
  const double q = q_percent / 100.0;
  const double virtual_index = static_cast<double>(n - 1) * q;
  const double prev_floor = std::floor(virtual_index);
  std::size_t prev = 0;
  std::size_t next = 0;
  double previous_index = prev_floor;
  if (virtual_index >= static_cast<double>(n - 1)) {
    prev = next = n - 1;
    previous_index = -1.0;  // NumPy marks out-of-range indexes as -1 before computing gamma
  } else if (virtual_index < 0.0) {
    prev = next = 0;
    previous_index = 0.0;
  } else {
    prev = static_cast<std::size_t>(prev_floor);
    next = prev + 1;
  }
  const double gamma = virtual_index - previous_index;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(prev), values.end());
  const double a = values[prev];
  double b = a;
  if (next != prev) {
    b = *std::min_element(values.begin() + static_cast<std::ptrdiff_t>(prev) + 1, values.end());
  }
  return lerp(a, b, gamma);
}

}  // namespace paudit::numeric
