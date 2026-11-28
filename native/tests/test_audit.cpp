#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/pyformat.hpp"
#include "paudit/audit.hpp"

using paudit::detail::py_fixed;
using paudit::detail::py_general;
using paudit::detail::py_percent0;

TEST_CASE("number formatting matches Python f-strings", "[audit][format]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // Expected strings captured from CPython: f"{v:.3f}", f"{v:+.4f}", f"{v:.3g}", f"{v:.4g}", f"{v:g}",
  // f"{v:.0%}"
  struct Row {
    double v;
    const char* f3;
    const char* plus4;
    const char* g3;
    const char* g4;
    const char* g;
    const char* pct;
  };
  const std::vector<Row> rows{
      {0.0, "0.000", "+0.0000", "0", "0", "0", "0%"},
      {-0.0, "-0.000", "-0.0000", "-0", "-0", "-0", "-0%"},
      {0.12345, "0.123", "+0.1235", "0.123", "0.1235", "0.12345", "12%"},
      {1234567.0, "1234567.000", "+1234567.0000", "1.23e+06", "1.235e+06", "1.23457e+06", "123456700%"},
      {1e-5, "0.000", "+0.0000", "1e-05", "1e-05", "1e-05", "0%"},
      {0.000123456, "0.000", "+0.0001", "0.000123", "0.0001235", "0.000123456", "0%"},
      {0.125, "0.125", "+0.1250", "0.125", "0.125", "0.125", "12%"},
      {99.995, "99.995", "+99.9950", "100", "100", "99.995", "10000%"},
      {nan, "nan", "+nan", "nan", "nan", "nan", "nan%"},
      {-nan, "nan", "+nan", "nan", "nan", "nan", "nan%"},
      {inf, "inf", "+inf", "inf", "inf", "inf", "inf%"},
      {-inf, "-inf", "-inf", "-inf", "-inf", "-inf", "-inf%"},
      {-1.23456e-7, "-0.000", "-0.0000", "-1.23e-07", "-1.235e-07", "-1.23456e-07", "-0%"},
  };
  for (const auto& r : rows) {
    CAPTURE(r.v);
    CHECK(py_fixed(r.v, 3) == r.f3);
    CHECK(py_fixed(r.v, 4, true) == r.plus4);
    CHECK(py_general(r.v, 3) == r.g3);
    CHECK(py_general(r.v, 4) == r.g4);
    CHECK(py_general(r.v) == r.g);
    CHECK(py_percent0(r.v) == r.pct);
  }
}

namespace {

struct Frame {
  std::vector<double> y;
  std::vector<double> pred;
  std::vector<std::int64_t> t;
  std::vector<std::uint8_t> split;
};

Frame make_frame(std::size_t n) {
  Frame f;
  for (std::size_t i = 0; i < n; ++i) {
    const auto x = static_cast<double>(i);
    f.y.push_back(std::sin(x / 5.0) + 0.1 * std::sin(x * 7.1));
    f.pred.push_back(std::sin(x / 5.0));
    f.t.push_back(static_cast<std::int64_t>(n - i));  // reversed: the pipeline must sort
    f.split.push_back(i < n / 3 ? paudit::kSplitTest : paudit::kSplitTrain);
  }
  return f;
}

paudit::AuditColumns columns(const Frame& f) {
  paudit::AuditColumns c;
  c.target = f.y;
  c.pred = std::span<const double>(f.pred);
  c.time = paudit::KeyColumn(std::span<const std::int64_t>(f.t));
  c.split = std::span<const std::uint8_t>(f.split);
  return c;
}

paudit::AuditConfig config() {
  paudit::AuditConfig cfg;
  cfg.target_name = "y";
  cfg.model_name = "pred";
  cfg.split_name = "split";
  cfg.n_boot = 50;
  return cfg;
}

}  // namespace

TEST_CASE("audit sorts by time and runs every check", "[audit]") {
  const Frame f = make_frame(300);
  const auto report = paudit::audit(columns(f), config());
  CHECK(report.n_eval > 50);
  REQUIRE(report.checks.size() == 3);  // label memory, beats baselines, temporal split
  CHECK(report.checks[0].name == "label memory");
  CHECK(report.checks[1].name == "beats trivial baselines");
  CHECK(report.checks[2].name == "temporal split");
  CHECK(report.checks[2].status == paudit::Status::pass);  // test rows come last once sorted by time
  CHECK(report.table.front().first == "pred");
}

TEST_CASE("data errors carry the Python messages", "[audit]") {
  Frame f = make_frame(60);
  auto cfg = config();
  SECTION("no test rows") {
    std::fill(f.split.begin(), f.split.end(), paudit::kSplitTrain);
    CHECK_THROWS_WITH(paudit::audit(columns(f), cfg), "no rows with split == 'test'");
  }
  SECTION("non-binary classification target") {
    cfg.task = paudit::Task::classification;
    cfg.metric = paudit::Metric::f1;
    CHECK_THROWS_WITH(paudit::audit(columns(f), cfg), "classification target must be binary 0/1");
  }
  SECTION("too few scored rows") {
    for (std::size_t i = 0; i < f.split.size(); ++i)
      f.split[i] = i < 5 ? paudit::kSplitTest : paudit::kSplitTrain;
    CHECK_THROWS_WITH(paudit::audit(columns(f), cfg),
                      "only 5 rows can be scored; need test rows with enough history");
  }
}
