#include "paudit/csv.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paudit/parallel.hpp"

// fast_float is included as a system header (see CMakeLists.txt): pandas >= 3.1 parses floats with it.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <fast_float/fast_float.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace paudit::io {

std::size_t Column::size() const noexcept {
  switch (kind) {
    case ColumnKind::int64:
    case ColumnKind::boolean:
      return ints.size();
    case ColumnKind::float64:
      return floats.size();
    case ColumnKind::string:
      return strings.size();
  }
  return 0;
}

const Column* Table::find(std::string_view name) const noexcept {
  for (const auto& c : columns) {
    if (c.name == name) return &c;
  }
  return nullptr;
}

namespace {

constexpr bool is_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}
constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
constexpr char lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (lower(a[i]) != lower(b[i])) return false;
  }
  return true;
}

// pandas accumulates mantissa digits with `number = number * 10. + digit`. Its macOS arm64 wheels
// are built by Apple Clang, which contracts that line into one fused multiply-add; its Linux
// (x86-64 and aarch64, GCC) and Windows wheels round twice. (Checked against pandas 2.3 and 3.0.)
// The two differ in the last bit for ~6% of 17-digit inputs, so follow this platform's wheels.
// This library itself is built with -ffp-contract=off, so the plain branch is never fused.
double accumulate_digit(double number, char digit) noexcept {
  const auto d = static_cast<double>(digit - '0');
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  return std::fma(number, 10.0, d);
#else
  return number * 10.0 + d;
#endif
}

const std::array<double, 309>& powers_of_ten() {
  static const std::array<double, 309> table = [] {
    std::array<double, 309> t{};
    for (std::size_t i = 0; i < t.size(); ++i) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "1e%zu", i);
      t[i] = std::strtod(buf, nullptr);  // correctly rounded, like the C literals pandas uses
    }
    return t;
  }();
  return table;
}

// Port of pandas' precise_xstrtod (pandas <= 3.0, pandas/_libs/src/parser/tokenizer.c,
// BSD-3-Clause), the converter read_csv used by default: at most 17 significant digits (leading
// zeros included) accumulated in a double, then one multiply or divide by a power of ten.
std::optional<double> precise_xstrtod_v30(std::string_view s) noexcept {
  const char* p = s.data();
  const char* const end = p + s.size();
  while (p < end && is_space(*p)) ++p;
  bool negative = false;
  if (p < end && (*p == '-' || *p == '+')) {
    negative = *p == '-';
    ++p;
  }
  constexpr int kMaxDigits = 17;
  double number = 0.0;
  long long exponent = 0;
  int num_digits = 0;
  while (p < end && is_digit(*p)) {
    if (num_digits < kMaxDigits) {
      number = accumulate_digit(number, *p);
      ++num_digits;
    } else {
      ++exponent;
    }
    ++p;
  }
  if (p < end && *p == '.') {
    ++p;
    int num_decimals = 0;
    while (num_digits < kMaxDigits && p < end && is_digit(*p)) {
      number = accumulate_digit(number, *p);
      ++p;
      ++num_digits;
      ++num_decimals;
    }
    if (num_digits >= kMaxDigits) {
      while (p < end && is_digit(*p)) ++p;
    }
    exponent -= num_decimals;
  }
  if (num_digits == 0) return std::nullopt;
  if (negative) number = -number;

  if (p < end && (*p == 'e' || *p == 'E')) {
    // strtol: optional whitespace and sign, then digits. With no digits the marker stays
    // unconsumed and the field is rejected as trailing garbage below.
    const char* q = p + 1;
    while (q < end && is_space(*q)) ++q;
    bool exp_negative = false;
    if (q < end && (*q == '+' || *q == '-')) {
      exp_negative = *q == '-';
      ++q;
    }
    const char* const digits = q;
    long long n = 0;
    while (q < end && is_digit(*q)) {
      if (n < 100'000'000) n = n * 10 + (*q - '0');  // beyond this every result is 0 or inf
      ++q;
    }
    if (q != digits) {
      exponent += exp_negative ? -n : n;
      p = q;
    }
  }

  const auto& e = powers_of_ten();
  if (exponent > 308) {
    number = number == 0.0 ? 0.0 : (number > 0.0 ? HUGE_VAL : -HUGE_VAL);
  } else if (exponent > 0) {
    number *= e[static_cast<std::size_t>(exponent)];
  } else if (exponent < -308) {
    if (exponent < -616) {
      number = 0.0;
    } else {
      number /= e[static_cast<std::size_t>(-308 - exponent)];
      number /= e[308];
    }
  } else {
    number /= e[static_cast<std::size_t>(-exponent)];
  }
  while (p < end && is_space(*p)) ++p;
  if (p != end) return std::nullopt;
  return number;
}

// --- pandas >= 3.1 (pandas/_libs/src/parser/tokenizer.cpp and fast_float_wrappers.cpp) ---------

/// End of the longest prefix of s[p, end) matching [sign](\d+\.?\d*|\.\d+)([eE][+-]?\d+)?,
/// or npos. `plus` allows a leading '+'.
std::size_t scan_number(std::string_view s, std::size_t p, std::size_t end, bool plus) noexcept {
  std::size_t i = p;
  if (i < end && (s[i] == '-' || (plus && s[i] == '+'))) ++i;
  const std::size_t int_start = i;
  while (i < end && is_digit(s[i])) ++i;
  const bool int_digits = i > int_start;
  if (i < end && s[i] == '.') {
    const std::size_t frac_start = i + 1;
    std::size_t j = frac_start;
    while (j < end && is_digit(s[j])) ++j;
    if (int_digits || j > frac_start) i = j;
  }
  if (!int_digits && (i == int_start || s[int_start] != '.')) return std::string_view::npos;
  if (i < end && (s[i] == 'e' || s[i] == 'E')) {
    std::size_t j = i + 1;
    if (j < end && (s[j] == '+' || s[j] == '-')) ++j;
    const std::size_t exp_start = j;
    while (j < end && is_digit(s[j])) ++j;
    if (j > exp_start) i = j;
  }
  return i;
}

/// fast_float::from_chars on a span already known to be a plain number (correctly rounded;
/// out-of-range values become +-inf or +-0, which pandas accepts).
double fast_parse(std::string_view number) noexcept {
  bool negative = false;
  if (!number.empty() && (number.front() == '+' || number.front() == '-')) {
    negative = number.front() == '-';
    number.remove_prefix(1);
  }
  double v = 0.0;
  (void)fast_float::from_chars(number.data(), number.data() + number.size(), v);
  return negative ? -v : v;
}

/// The fallback pandas 3.1 uses when fast_float declines (a '+' sign after whitespace, ...):
/// up to 17 significant digits (leading zeros skipped) in an integer mantissa, one rounding to
/// double, then the same power-of-ten scaling as before.
std::optional<double> fallback_v31(std::string_view s) noexcept {
  const char* p = s.data();
  const char* const end = p + s.size();
  while (p < end && is_space(*p)) ++p;
  bool negative = false;
  if (p < end && (*p == '-' || *p == '+')) {
    negative = *p == '-';
    ++p;
  }
  constexpr int kMaxDigits = 17;
  bool saw_digit = false;
  std::uint64_t mantissa = 0;
  int num_digits = 0;
  long long exponent = 0;
  while (p < end && *p == '0') {
    saw_digit = true;
    ++p;
  }
  while (p < end && is_digit(*p)) {
    if (num_digits < kMaxDigits) {
      mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
      ++num_digits;
    } else {
      ++exponent;
    }
    ++p;
  }
  if (p < end && *p == '.') {
    ++p;
    long long num_decimals = 0;
    if (num_digits == 0) {
      while (p < end && *p == '0') {
        saw_digit = true;
        ++num_decimals;
        ++p;
      }
    }
    while (num_digits < kMaxDigits && p < end && is_digit(*p)) {
      mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
      ++num_digits;
      ++num_decimals;
      ++p;
    }
    if (num_digits >= kMaxDigits) {
      while (p < end && is_digit(*p)) ++p;
    }
    exponent -= num_decimals;
  }
  if (num_digits == 0 && !saw_digit) return std::nullopt;
  double number = static_cast<double>(mantissa);
  if (negative) number = -number;
  if (p < end && (*p == 'e' || *p == 'E')) {  // strtol: optional whitespace and sign, then digits
    const char* q = p + 1;
    while (q < end && is_space(*q)) ++q;
    bool exp_negative = false;
    if (q < end && (*q == '+' || *q == '-')) {
      exp_negative = *q == '-';
      ++q;
    }
    const char* const digits = q;
    long long n = 0;
    while (q < end && is_digit(*q)) {
      if (n < 100'000'000) n = n * 10 + (*q - '0');
      ++q;
    }
    if (q != digits) {
      exponent += exp_negative ? -n : n;
      p = q;
    }
  }
  const auto& e = powers_of_ten();
  if (exponent > 308) {
    number = number == 0.0 ? 0.0 : (number < 0.0 ? -HUGE_VAL : HUGE_VAL);
  } else if (exponent > 0) {
    number *= e[static_cast<std::size_t>(exponent)];
  } else if (exponent < -308) {
    if (exponent < -616) {
      number = 0.0;
    } else {
      number /= e[static_cast<std::size_t>(-308 - exponent)];
      number /= e[308];
    }
  } else {
    number /= e[static_cast<std::size_t>(-exponent)];
  }
  while (p < end && is_space(*p)) ++p;
  if (p != end) return std::nullopt;
  return number;
}

std::optional<double> precise_xstrtod_v31(std::string_view s) noexcept {
  const std::size_t n = s.size();
  // try_parse_plain_double: the whole field is a plain number (a '+' allowed, no whitespace).
  if (scan_number(s, 0, n, true) == n && n > 0) return fast_parse(s);
  // precise_xstrtod: whitespace, then fast_float on the token, else the fallback.
  std::size_t p = 0;
  while (p < n && is_space(s[p])) ++p;
  const std::size_t q = p + ((p < n && (s[p] == '+' || s[p] == '-')) ? 1 : 0);
  const bool starts_number = q < n && (is_digit(s[q]) || (s[q] == '.' && q + 1 < n && is_digit(s[q + 1])));
  if (starts_number && s[p] != '+') {
    std::size_t token_end = p;
    while (token_end < n && !is_space(s[token_end])) ++token_end;
    std::size_t r = scan_number(s, p, token_end, false);
    const double v = fast_parse(s.substr(p, r - p));
    while (r < n && is_space(s[r])) ++r;
    if (r != n) return std::nullopt;
    return v;
  }
  return fallback_v31(s);
}

constexpr std::array<std::string_view, 19> kNaValues = {
    "",     "#N/A", "#N/A N/A", "#NA",  "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A",  "NA",       "NULL", "NaN",     "None",     "n/a",  "nan",  "null"};

bool is_true(std::string_view s) noexcept { return iequals(s, "true"); }
bool is_false(std::string_view s) noexcept { return iequals(s, "false"); }

std::string quoted(std::string_view name) {
  std::string out = "'";
  out += name;
  out += "'";
  return out;
}

bool valid_utf8(std::string_view s) noexcept {
  std::size_t i = 0;
  const std::size_t n = s.size();
  while (i < n) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      ++i;
      continue;
    }
    std::size_t len = 0;
    std::uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) {
      len = 2;
      cp = c & 0x1Fu;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
      cp = c & 0x0Fu;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
      cp = c & 0x07u;
    } else {
      return false;
    }
    if (i + len > n) return false;
    for (std::size_t k = 1; k < len; ++k) {
      const auto cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (cc & 0x3Fu);
    }
    const bool overlong = (len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000);
    if (overlong || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    i += len;
  }
  return true;
}

struct Tokens {
  std::vector<std::string> header;
  std::vector<std::string_view> fields;  ///< row-major, header.size() per row
  std::size_t n_rows = 0;
  std::deque<std::string> arena;  ///< unescaped quoted fields (stable addresses)
};

// pandas' C tokenizer with the default dialect: '"' quotes a field only at its start, a doubled
// quote inside a quoted field is a literal quote, and text after the closing quote is appended.
Tokens tokenize(std::string_view text, char sep) {
  Tokens t;
  const std::size_t n = text.size();
  std::size_t pos = 0;
  std::size_t records = 0;  // records so far, blank lines included: pandas' "line" numbers
  std::size_t n_cols = 0;
  bool have_header = false;
  std::vector<std::string_view> row;

  auto at_line_end = [&](std::size_t i) { return i >= n || text[i] == '\n' || text[i] == '\r'; };
  auto skip_line_end = [&] {
    if (pos < n && text[pos] == '\r') {
      ++pos;
      if (pos < n && text[pos] == '\n') {
        ++pos;
      } else if (pos < n) {
        // pandas' tokenizer has bugs after a bare CR line ending: a following line that starts
        // with whitespace repeats earlier fields (or reports a buffer overflow), and after a blank
        // line a leading empty field is dropped. Rather than copy them, leave such files to pandas.
        throw Unsupported("bare carriage-return (classic Mac) line endings");
      }
    } else if (pos < n && text[pos] == '\n') {
      ++pos;
    }
  };

  while (pos < n) {
    // Blank and whitespace-only lines are skipped (but still counted).
    std::size_t q = pos;
    while (q < n && (text[q] == ' ' || text[q] == '\t') && text[q] != sep) ++q;
    if (at_line_end(q)) {
      pos = q;
      skip_line_end();
      ++records;
      continue;
    }

    row.clear();
    for (;;) {
      std::string_view field;
      if (pos < n && text[pos] == '"') {
        const std::size_t inner = ++pos;
        std::string* buf = nullptr;
        std::size_t seg = inner;
        for (;;) {
          const std::size_t close = text.find('"', pos);
          if (close == std::string_view::npos) {
            throw ParseError("Error tokenizing data. C error: EOF inside string starting at row " +
                             std::to_string(records));
          }
          if (close + 1 < n && text[close + 1] == '"') {  // "" -> "
            if (buf == nullptr) buf = &t.arena.emplace_back();
            buf->append(text.substr(seg, close + 1 - seg));
            pos = close + 2;
            seg = pos;
            continue;
          }
          std::size_t tail = close + 1;
          std::size_t tail_end = tail;
          while (tail_end < n && text[tail_end] != sep && !at_line_end(tail_end)) ++tail_end;
          if (buf == nullptr && tail_end == tail) {
            field = text.substr(inner, close - inner);
          } else {
            if (buf == nullptr) buf = &t.arena.emplace_back();
            buf->append(text.substr(seg, close - seg));
            buf->append(text.substr(tail, tail_end - tail));
            field = *buf;
          }
          pos = tail_end;
          break;
        }
      } else {
        std::size_t e = pos;
        while (e < n && text[e] != sep && !at_line_end(e)) ++e;
        field = text.substr(pos, e - pos);
        pos = e;
      }
      row.push_back(field);
      if (pos < n && text[pos] == sep) {
        ++pos;
        if (pos >= n) {  // trailing separator at end of file: one more, empty field
          row.emplace_back();
          break;
        }
        continue;
      }
      break;
    }
    skip_line_end();
    ++records;

    if (!have_header) {
      t.header.assign(row.begin(), row.end());
      n_cols = row.size();
      have_header = true;
      continue;
    }
    if (row.size() > n_cols) {
      if (t.n_rows == 0) {
        throw Unsupported(
            "the first data row has more fields than the header (pandas would read the extra "
            "leading fields as an index)");
      }
      throw ParseError("Error tokenizing data. C error: Expected " + std::to_string(n_cols) +
                       " fields in line " + std::to_string(records) + ", saw " + std::to_string(row.size()) +
                       "\n");
    }
    t.fields.insert(t.fields.end(), row.begin(), row.end());
    t.fields.resize(t.fields.size() + (n_cols - row.size()));  // short rows: missing values
    ++t.n_rows;
  }
  if (!have_header) throw ParseError("No columns to parse from file");
  return t;
}

// pandas' header clean-up: empty names become "Unnamed: i", and repeated names get ".1", ".2",
// ... suffixes, skipping suffixed names that already appear in the header.
std::vector<std::string> clean_header(std::vector<std::string> names) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i].empty()) names[i] = "Unnamed: " + std::to_string(i);
  }
  const std::unordered_set<std::string> original(names.begin(), names.end());
  std::unordered_map<std::string, std::size_t> counts;
  for (auto& item : names) {
    const std::string old_item = item;
    std::size_t cur = counts.contains(item) ? counts[item] : 0;
    while (cur > 0) {
      counts[old_item] = cur + 1;
      item = old_item + "." + std::to_string(cur);
      if (original.contains(item)) {
        ++cur;
      } else {
        cur = counts.contains(item) ? counts[item] : 0;
      }
    }
    counts[item] = cur + 1;
  }
  return names;
}

// pandas' C-parser type inference for one column: int64 if every value is an integer and none
// is missing, else float64 if every present value is a number, else bool if every present value
// is true/false (any case), else strings.
Column convert(std::string name, const Tokens& t, std::size_t col, Pandas pandas) {
  Column c;
  c.name = std::move(name);
  const std::size_t n = t.n_rows;
  const std::size_t stride = t.header.size();
  auto field = [&](std::size_t r) { return t.fields[r * stride + col]; };
  if (n == 0) return c;  // header only: pandas gives an empty object column

  std::vector<std::uint8_t> na(n, 0);
  bool any_na = false;
  bool all_int = true;
  std::vector<std::int64_t> ints(n, 0);
  for (std::size_t r = 0; r < n; ++r) {
    const auto f = field(r);
    if (is_default_na(f)) {
      na[r] = 1;
      any_na = true;
      continue;
    }
    bool overflow = false;
    const auto v = parse_pandas_int64(f, overflow);
    if (overflow) {
      throw Unsupported("column " + quoted(c.name) + " has integers outside the int64 range");
    }
    if (v) {
      ints[r] = *v;
    } else {
      all_int = false;
    }
  }
  if (all_int && !any_na) {
    c.kind = ColumnKind::int64;
    c.ints = std::move(ints);
    return c;
  }
  if (all_int) {
    // Integers with missing values: pandas parses them as int64, then casts to float64 (so a
    // value above 2^53 rounds once, correctly, not through the float converter). Before 3.1
    // pandas marked the missing slots with INT64_MIN, so a real INT64_MIN came out as NaN too.
    c.kind = ColumnKind::float64;
    c.floats.assign(n, std::numeric_limits<double>::quiet_NaN());
    const bool min_is_na = pandas == Pandas::v2_v30;
    for (std::size_t r = 0; r < n; ++r) {
      if (na[r] == 0 && !(min_is_na && ints[r] == std::numeric_limits<std::int64_t>::min())) {
        c.floats[r] = static_cast<double>(ints[r]);
      }
    }
    return c;
  }

  std::vector<double> floats(n, std::numeric_limits<double>::quiet_NaN());
  bool all_float = true;
  for (std::size_t r = 0; r < n && all_float; ++r) {
    if (na[r] != 0) continue;
    const auto v = parse_pandas_double(field(r), pandas);
    if (v) {
      floats[r] = *v;
    } else {
      all_float = false;
    }
  }
  if (all_float) {
    c.kind = ColumnKind::float64;
    c.floats = std::move(floats);
    return c;
  }

  bool all_bool = true;
  for (std::size_t r = 0; r < n && all_bool; ++r) {
    if (na[r] == 0 && !is_true(field(r)) && !is_false(field(r))) all_bool = false;
  }
  if (all_bool) {
    if (!any_na) {
      c.kind = ColumnKind::boolean;
      c.ints.resize(n);
      for (std::size_t r = 0; r < n; ++r) c.ints[r] = is_true(field(r)) ? 1 : 0;
    } else {
      // pandas: an object column of True/False/NaN. Every use the CLI makes of it (as a number,
      // a split label, or a key, which rejects missing values) matches a 1/0/NaN float column.
      c.kind = ColumnKind::float64;
      c.floats.assign(n, std::numeric_limits<double>::quiet_NaN());
      for (std::size_t r = 0; r < n; ++r) {
        if (na[r] == 0) c.floats[r] = is_true(field(r)) ? 1.0 : 0.0;
      }
    }
    return c;
  }

  c.kind = ColumnKind::string;
  c.strings.resize(n);
  c.missing = std::move(na);
  for (std::size_t r = 0; r < n; ++r) {
    if (c.missing[r] == 0) c.strings[r] = std::string(field(r));
  }
  return c;
}

}  // namespace

bool is_default_na(std::string_view field) noexcept {
  return std::find(kNaValues.begin(), kNaValues.end(), field) != kNaValues.end();
}

std::optional<double> parse_pandas_double(std::string_view field, Pandas pandas) noexcept {
  if (const auto v = pandas == Pandas::v31 ? precise_xstrtod_v31(field) : precise_xstrtod_v30(field))
    return v;
  if (iequals(field, "inf") || iequals(field, "+inf") || iequals(field, "infinity") ||
      iequals(field, "+infinity")) {
    return HUGE_VAL;
  }
  if (iequals(field, "-inf") || iequals(field, "-infinity")) return -HUGE_VAL;
  return std::nullopt;
}

std::optional<std::int64_t> parse_pandas_int64(std::string_view field, bool& overflow) noexcept {
  overflow = false;
  const char* p = field.data();
  const char* const end = p + field.size();
  while (p < end && is_space(*p)) ++p;
  bool negative = false;
  if (p < end && (*p == '-' || *p == '+')) {
    negative = *p == '-';
    ++p;
  }
  if (p >= end || !is_digit(*p)) return std::nullopt;
  const std::uint64_t limit = negative ? std::uint64_t{1} << 63 : (std::uint64_t{1} << 63) - 1;
  std::uint64_t acc = 0;
  bool too_big = false;
  while (p < end && is_digit(*p)) {
    const auto d = static_cast<std::uint64_t>(*p - '0');
    if (!too_big && acc > (limit - d) / 10) too_big = true;
    if (!too_big) acc = acc * 10 + d;
    ++p;
  }
  if (too_big) {
    // strtoll stops at the digits; pandas reports overflow only if nothing follows them,
    // otherwise the field is merely "not an integer" and gets another chance as a float.
    overflow = p == end;
    return std::nullopt;
  }
  while (p < end && is_space(*p)) ++p;
  if (p != end) return std::nullopt;
  if (negative)
    return acc == (std::uint64_t{1} << 63) ? std::numeric_limits<std::int64_t>::min()
                                           : -static_cast<std::int64_t>(acc);
  return static_cast<std::int64_t>(acc);
}

Table read_csv(std::string_view text, char sep, Pandas pandas) {
  if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);
  if (!valid_utf8(text)) throw Unsupported("the file is not valid UTF-8");
  Tokens tokens = tokenize(text, sep);
  std::vector<std::string> names = clean_header(tokens.header);

  Table table;
  table.n_rows = tokens.n_rows;
  table.columns.resize(names.size());
  std::vector<std::exception_ptr> errors(names.size());
  parallel::parallel_for(names.size(), 1, [&](std::size_t begin, std::size_t end) {
    for (std::size_t c = begin; c < end; ++c) {
      try {
        table.columns[c] = convert(std::move(names[c]), tokens, c, pandas);
      } catch (...) {
        errors[c] = std::current_exception();
      }
    }
  });
  for (const auto& e : errors) {  // the leftmost column's error, whatever the thread schedule
    if (e) std::rethrow_exception(e);
  }
  return table;
}

}  // namespace paudit::io
