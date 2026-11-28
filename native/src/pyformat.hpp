#pragma once

// Python f-string number formatting ({x:.3f}, {x:+.4f}, {x:.3g}, {x:g}, {x:.0%}) for the report
// summaries. Finite values go through snprintf, which is correctly rounded like Python's own
// formatter; NaN and infinities are spelled the way Python spells them on every platform.

#include <cmath>
#include <cstdio>
#include <string>

namespace paudit::detail {

/// "nan" / "inf" / "-inf", with a forced '+' for positive values (and NaN) when `plus` is set.
[[nodiscard]] inline std::string py_nonfinite(double x, bool plus) {
  if (std::isnan(x)) return plus ? "+nan" : "nan";
  if (x > 0) return plus ? "+inf" : "inf";
  return "-inf";
}

/// f"{x:.{prec}f}" (or "+.{prec}f" when `plus`).
[[nodiscard]] inline std::string py_fixed(double x, int prec, bool plus = false) {
  if (!std::isfinite(x)) return py_nonfinite(x, plus);
  char buf[512];  // %f of the largest double has ~310 integer digits
  const int len = plus ? std::snprintf(buf, sizeof(buf), "%+.*f", prec, x)
                       : std::snprintf(buf, sizeof(buf), "%.*f", prec, x);
  return std::string(buf, static_cast<std::size_t>(len > 0 ? len : 0));
}

/// f"{x:.{prec}g}"; Python's bare "{x:g}" is precision 6.
[[nodiscard]] inline std::string py_general(double x, int prec = 6) {
  if (!std::isfinite(x)) return py_nonfinite(x, false);
  char buf[64];
  const int len = std::snprintf(buf, sizeof(buf), "%.*g", prec, x);
  return std::string(buf, static_cast<std::size_t>(len > 0 ? len : 0));
}

/// f"{x:.0%}": x * 100 in fixed notation with no decimals, then '%'.
[[nodiscard]] inline std::string py_percent0(double x) { return py_fixed(x * 100.0, 0) + "%"; }

}  // namespace paudit::detail
