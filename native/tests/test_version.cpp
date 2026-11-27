#include <catch2/catch_test_macros.hpp>

#include <string_view>

#include "paudit/version.hpp"

TEST_CASE("version is a dotted triple", "[version]") {
  const std::string_view v = paudit::version();
  REQUIRE_FALSE(v.empty());
  int dots = 0;
  for (const char c : v) {
    if (c == '.') {
      ++dots;
    } else {
      REQUIRE((c >= '0' && c <= '9'));
    }
  }
  REQUIRE(dots == 2);
}

TEST_CASE("build info is consistent", "[version]") {
  const auto info = paudit::build_info();
  REQUIRE(info.version == paudit::version());
  REQUIRE(info.cplusplus >= 202002L);
  REQUIRE_FALSE(info.compiler.empty());
}
