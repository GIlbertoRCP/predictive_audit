#pragma once

// Python f-string number formatting ({x:.3f}, {x:+.4f}, {x:.3g}, {x:g}, {x:.0%}) for the report
// summaries. Finite values go through snprintf, which is correctly rounded like Python's own
// formatter; NaN and infinities are spelled the way Python spells them on every platform.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

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

/// repr(x) for a float: the shortest digits that read back as x, in fixed notation for decimal
/// exponents -4..15 (with ".0" for whole numbers) and as d.ddde+XX otherwise.
/// (std::to_chars would give the digits, but needs macOS 13.3+ for floating point.)
[[nodiscard]] inline std::string py_repr(double x) {
  if (!std::isfinite(x)) return py_nonfinite(x, false);
  if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
  // Shortest digits that read back as x. The correctly rounded p-digit string usually does when
  // any p-digit string does, except next to a power of two, where the gap below x is half the
  // gap above: then its neighbour one unit away on the wide side is the answer.
  char buf[48];
  const auto with_digits = [&](std::uint64_t m, int prec, int exp10) {
    std::string digits = std::to_string(m);
    if (static_cast<int>(digits.size()) != prec) return false;  // carried or lost a digit
    std::snprintf(buf, sizeof(buf), "%s%c%s%se%d", std::signbit(x) ? "-" : "", digits[0], prec > 1 ? "." : "",
                  digits.c_str() + 1, exp10);
    return std::strtod(buf, nullptr) == x;
  };
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
    if (std::strtod(buf, nullptr) == x) break;
    const std::string sci_try(buf);
    const std::size_t e_at = sci_try.find('e');
    std::string mant;
    for (std::size_t k = 0; k < e_at; ++k) {
      if (sci_try[k] >= '0' && sci_try[k] <= '9') mant += sci_try[k];
    }
    const int exp_try = std::atoi(sci_try.c_str() + e_at + 1);
    const std::uint64_t m = std::stoull(mant);
    if (with_digits(m + 1, prec, exp_try) || (m > 0 && with_digits(m - 1, prec, exp_try))) break;
  }
  const std::string sci(buf);
  const std::size_t e_pos = sci.find('e');
  std::string mantissa = sci.substr(0, e_pos);
  const int exp10 = std::atoi(sci.c_str() + e_pos + 1);
  const bool negative = mantissa.front() == '-';
  std::string digits;
  for (const char c : mantissa) {
    if (c >= '0' && c <= '9') digits += c;
  }
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();

  std::string out = negative ? "-" : "";
  if (exp10 >= -4 && exp10 < 16) {
    if (exp10 >= 0) {
      const auto int_len = static_cast<std::size_t>(exp10) + 1;
      if (digits.size() <= int_len) {
        out += digits + std::string(int_len - digits.size(), '0') + ".0";
      } else {
        out += digits.substr(0, int_len) + "." + digits.substr(int_len);
      }
    } else {
      out += "0." + std::string(static_cast<std::size_t>(-exp10 - 1), '0') + digits;
    }
    return out;
  }
  out += digits.substr(0, 1);
  if (digits.size() > 1) out += "." + digits.substr(1);
  char exp_buf[16];
  std::snprintf(exp_buf, sizeof(exp_buf), "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
  return out + exp_buf;
}

/// len(s) for a UTF-8 string: the number of code points.
[[nodiscard]] inline std::size_t py_len(std::string_view s) noexcept {
  std::size_t n = 0;
  for (const char c : s) n += (static_cast<unsigned char>(c) & 0xC0) != 0x80 ? 1 : 0;
  return n;
}

/// repr(s) for a str: single quotes unless the text holds a ' and no ", with backslash escapes
/// for the quote, backslash, and control characters.
[[nodiscard]] inline std::string py_repr_str(std::string_view s) {
  const bool use_double = s.find('\'') != std::string_view::npos && s.find('"') == std::string_view::npos;
  const char q = use_double ? '"' : '\'';
  std::string out(1, q);
  for (const char c : s) {
    const auto u = static_cast<unsigned char>(c);
    if (c == q || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else if (u < 0x20 || u == 0x7f) {
      char hex[8];
      std::snprintf(hex, sizeof(hex), "\\x%02x", u);
      out += hex;
    } else {
      out += c;
    }
  }
  out += q;
  return out;
}

}  // namespace paudit::detail
