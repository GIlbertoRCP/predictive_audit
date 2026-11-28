#pragma once

// A CSV/TSV reader that produces the same columns as `pandas.read_csv(path)` / `(path, sep="\t")`
// with default options, for the inputs the standalone CLI accepts:
// - the same column types: int64, float64, bool, or string, inferred the way pandas' C parser
//   infers them;
// - pandas' default NA strings, quoting with doubled quotes, a UTF-8 BOM, and \n, \r\n or \r
//   line endings; blank and whitespace-only lines are skipped;
// - bit-identical floats: pandas' default converter is not (always) correctly rounded, so it is
//   ported. It changed in pandas 3.1, so the reader follows a given pandas version (`Pandas`);
// - pandas' header handling ("Unnamed: i" for empty names, ".1"/".2" suffixes for duplicates)
//   and pandas' error messages for empty input, ragged rows, and unterminated quotes.
//
// Inputs whose pandas result has no faithful equivalent here throw `Unsupported`, so the Python
// package can hand them to pandas instead (e.g. integers beyond int64, or a first data row longer
// than the header, which pandas silently turns into an index).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace paudit::io {

enum class ColumnKind : std::uint8_t { int64, float64, boolean, string };

/// Which pandas read_csv to reproduce: its float converter changed in 3.0 and again in 3.1.
enum class Pandas : std::uint8_t {
  v2,   ///< pandas 2.x: precise_xstrtod, overflow is an error, no space after 'e'
  v30,  ///< pandas 3.0: precise_xstrtod (17 digits in a double, FMA on macOS arm64)
  v31,  ///< pandas >= 3.1: fast_float (correctly rounded), with a rewritten fallback
};

/// The Pandas value for a pandas (major, minor) version.
[[nodiscard]] constexpr Pandas pandas_for_version(int major, int minor) noexcept {
  if (major < 3) return Pandas::v2;
  return (major == 3 && minor == 0) ? Pandas::v30 : Pandas::v31;
}

struct Column {
  std::string name;
  ColumnKind kind = ColumnKind::string;
  std::vector<std::int64_t> ints;     ///< int64 values, or 0/1 for boolean
  std::vector<double> floats;         ///< float64 values (NaN = missing)
  std::vector<std::string> strings;   ///< string values
  std::vector<std::uint8_t> missing;  ///< string columns: 1 where the value is missing

  [[nodiscard]] std::size_t size() const noexcept;
};

struct Table {
  std::vector<Column> columns;
  std::size_t n_rows = 0;

  [[nodiscard]] const Column* find(std::string_view name) const noexcept;
};

/// The input is valid but outside what the native reader reproduces; pandas handles it.
class Unsupported : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// An error pandas raises too, carrying pandas' message (EmptyDataError / ParserError).
class ParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Parse a whole file's contents. `sep` is ',' for CSV and '\t' for TSV.
[[nodiscard]] Table read_csv(std::string_view text, char sep, Pandas pandas = Pandas::v31);

/// pandas' default float conversion of one field (including its "inf"/"infinity" spellings),
/// or nullopt if pandas would not read the field as a number.
[[nodiscard]] std::optional<double> parse_pandas_double(std::string_view field,
                                                        Pandas pandas = Pandas::v31) noexcept;

/// pandas' int64 conversion of one field (strtoll semantics with surrounding whitespace).
/// `overflow` is set when the field is an integer that does not fit in int64.
[[nodiscard]] std::optional<std::int64_t> parse_pandas_int64(std::string_view field, bool& overflow) noexcept;

/// True for pandas' default NA strings ("", "NA", "NaN", "null", "#N/A", ...).
[[nodiscard]] bool is_default_na(std::string_view field) noexcept;

}  // namespace paudit::io
