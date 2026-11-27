#pragma once

// Internal FFT helpers on top of pocketfft (the same FFT library NumPy 2.x uses).

#include <cstddef>
#include <span>
#include <vector>

namespace paudit::detail {

/// `np.fft.irfft(|np.fft.rfft(x, n_fft)|^2, n_fft)`: the circular autocovariance of `x`
/// zero-padded to `n_fft` (unnormalized by len(x)), computed in NumPy's order of operations.
[[nodiscard]] std::vector<double> circular_autocov(std::span<const double> x, std::size_t n_fft);

}  // namespace paudit::detail
