#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "../src/pyformat.hpp"
#include "paudit/csv.hpp"

using paudit::io::ColumnKind;
using paudit::io::ParseError;
using paudit::io::read_csv;
using paudit::io::Unsupported;

namespace {

std::uint64_t bits(double x) {
  std::uint64_t b = 0;
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

}  // namespace

TEST_CASE("pandas 2.x/3.0 float conversion: precise_xstrtod", "[csv]") {
  const auto parse_pandas_double = [](std::string_view s) {
    return paudit::io::parse_pandas_double(s, paudit::io::Pandas::v2_v30);
  };
  CHECK(*parse_pandas_double("0.1") == 0.1);
  CHECK(*parse_pandas_double(" -2.5e3 ") == -2500.0);
  CHECK(*parse_pandas_double("1e 5") == 1e5);  // strtol skips the space
  CHECK(std::isinf(*parse_pandas_double("1e400")));
  CHECK(bits(*parse_pandas_double("-0e999")) == bits(0.0));   // zero mantissa: +0.0
  CHECK(bits(*parse_pandas_double("-1e-700")) == bits(0.0));  // below -616: +0.0
  CHECK(*parse_pandas_double("Infinity") == HUGE_VAL);
  CHECK(*parse_pandas_double("-INF") == -HUGE_VAL);
  CHECK_FALSE(parse_pandas_double(" inf"));
  CHECK_FALSE(parse_pandas_double("1.5E+"));
  CHECK_FALSE(parse_pandas_double("."));
  CHECK_FALSE(parse_pandas_double("1d5"));
  // Not correctly rounded (strtod gives ...674): 17 digits accumulated, then one division.
  CHECK(*parse_pandas_double("9107.2948605976750") == 9107.294860597676);
  // pandas' macOS arm64 wheels fuse the accumulation into FMAs; its other wheels round twice.
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  CHECK(*parse_pandas_double("6288.3607598386756") == 6288.360759838675);
#else
  CHECK(*parse_pandas_double("6288.3607598386756") == 6288.360759838676);
#endif
}

TEST_CASE("integer conversion follows strtoll with pandas' overflow rule", "[csv]") {
  using paudit::io::parse_pandas_int64;
  bool overflow = false;
  CHECK(*parse_pandas_int64(" +007 ", overflow) == 7);
  CHECK(*parse_pandas_int64("-9223372036854775808", overflow) == std::numeric_limits<std::int64_t>::min());
  CHECK_FALSE(parse_pandas_int64("9223372036854775808", overflow));
  CHECK(overflow);
  CHECK_FALSE(parse_pandas_int64("9223372036854775808 ", overflow));
  CHECK_FALSE(overflow);  // trailing text: "not an integer", not an overflow
  CHECK_FALSE(parse_pandas_int64("1.0", overflow));
}

TEST_CASE("column types are inferred like pandas", "[csv]") {
  const auto t = read_csv(
      "\xEF\xBB\xBFi,f,b,s,n,bn,big\r\n1,1.5,True,x,NA,true,1\r\n\r\n2,2,false,\"y,\"\"z\"\"\",,NA,NA\r\n",
      ',');
  REQUIRE(t.n_rows == 2);
  REQUIRE(t.columns.size() == 7);
  CHECK(t.columns[0].kind == ColumnKind::int64);
  CHECK(t.columns[1].kind == ColumnKind::float64);
  CHECK(t.columns[2].kind == ColumnKind::boolean);
  CHECK(t.columns[3].kind == ColumnKind::string);
  CHECK(t.columns[3].strings[1] == "y,\"z\"");
  CHECK(t.columns[4].kind == ColumnKind::float64);  // all missing
  CHECK(t.columns[5].kind == ColumnKind::float64);  // bool with NA: 1/0/NaN
  CHECK(t.columns[5].floats[0] == 1.0);
  CHECK(t.columns[6].kind == ColumnKind::float64);  // int with NA
}

TEST_CASE("header names are cleaned up like pandas", "[csv]") {
  const auto t = read_csv("a,a,a.1,\n1,2,3,4\n", ',');
  REQUIRE(t.columns.size() == 4);
  CHECK(t.columns[0].name == "a");
  CHECK(t.columns[1].name == "a.2");
  CHECK(t.columns[2].name == "a.1");
  CHECK(t.columns[3].name == "Unnamed: 3");
}

TEST_CASE("parse errors carry pandas' messages", "[csv]") {
  CHECK_THROWS_AS(read_csv("", ','), ParseError);
  CHECK_THROWS_WITH(read_csv("\n\n", ','), "No columns to parse from file");
  CHECK_THROWS_WITH(read_csv("a,b\n1,2\n\n3,4,5\n", ','),
                    "Error tokenizing data. C error: Expected 2 fields in line 4, saw 3\n");
  CHECK_THROWS_WITH(read_csv("a\n\"open\n", ','),
                    "Error tokenizing data. C error: EOF inside string starting at row 1");
  CHECK_THROWS_AS(read_csv("a,b\n1,2,3\n", ','), Unsupported);  // pandas: implicit index
  CHECK_THROWS_AS(read_csv("a\n99999999999999999999\n", ','), Unsupported);
  CHECK_THROWS_AS(read_csv("a\r1\r2\r", ','), Unsupported);  // bare CR line endings
}

TEST_CASE("JSON floats are spelled like Python's repr", "[report]") {
  using paudit::detail::py_repr;
  CHECK(py_repr(0.1) == "0.1");
  CHECK(py_repr(1.0) == "1.0");
  CHECK(py_repr(-0.0) == "-0.0");
  CHECK(py_repr(1e16) == "1e+16");
  CHECK(py_repr(9999999999999998.0) == "9999999999999998.0");
  CHECK(py_repr(1e-4) == "0.0001");
  CHECK(py_repr(1e-5) == "1e-05");
  CHECK(py_repr(5e-324) == "5e-324");
  CHECK(py_repr(1.7976931348623157e308) == "1.7976931348623157e+308");
  CHECK(py_repr(123456789.123) == "123456789.123");
  CHECK(paudit::detail::py_repr_str("it's") == "\"it's\"");
  CHECK(paudit::detail::py_len("caf\xC3\xA9") == 4);
}

TEST_CASE("pandas >= 3.1 float conversion: fast_float, then the rewritten fallback", "[csv]") {
  const auto parse = [](std::string_view s) {
    return paudit::io::parse_pandas_double(s, paudit::io::Pandas::v31);
  };
  CHECK(*parse("9107.2948605976750") == 9107.294860597674);  // correctly rounded now
  CHECK(*parse("000000000000000001.5") == 1.5);              // leading zeros don't use the digit budget
  CHECK(bits(*parse("-0e999")) == bits(-0.0));
  CHECK(bits(*parse("-1e-400")) == bits(-0.0));
  CHECK(std::isinf(*parse("1E400")));
  CHECK(*parse("+.5e-3") == 0.0005);
  CHECK(*parse("5.") == 5.0);
  CHECK_FALSE(parse("1e 5"));  // fast_float stops at the 'e'
  CHECK_FALSE(parse(" inf"));
  CHECK(*parse("-Infinity") == -HUGE_VAL);
  // A '+' with surrounding whitespace takes the fallback, which truncates to 17 digits.
  CHECK(*parse(" +16186104369156741856.326838832230") == 1.618610436915674e+19);
  CHECK(*parse("+16186104369156741856.326838832230") == 1.6186104369156741e+19);
}
