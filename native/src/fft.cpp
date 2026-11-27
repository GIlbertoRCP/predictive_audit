#include "fft.hpp"

#include <algorithm>
#include <complex>

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

namespace paudit::detail {

std::vector<double> circular_autocov(std::span<const double> x, std::size_t n_fft) {
  std::vector<double> padded(n_fft, 0.0);
  std::copy_n(x.begin(), std::min(x.size(), n_fft), padded.begin());
  std::vector<std::complex<double>> spectrum(n_fft / 2 + 1);
  const pocketfft::shape_t shape{n_fft};
  const pocketfft::stride_t real_stride{static_cast<std::ptrdiff_t>(sizeof(double))};
  const pocketfft::stride_t complex_stride{static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
  pocketfft::r2c(shape, real_stride, complex_stride, 0, pocketfft::FORWARD, padded.data(), spectrum.data(),
                 1.0);
  // f * conj(f) = (a*a - b*(-b)) + 0i, exactly what NumPy's complex multiply produces.
  for (auto& c : spectrum) c = std::complex<double>(c.real() * c.real() + c.imag() * c.imag(), 0.0);
  std::vector<double> out(n_fft);
  pocketfft::c2r(shape, complex_stride, real_stride, 0, pocketfft::BACKWARD, spectrum.data(), out.data(),
                 1.0 / static_cast<double>(n_fft));
  return out;
}

}  // namespace paudit::detail
