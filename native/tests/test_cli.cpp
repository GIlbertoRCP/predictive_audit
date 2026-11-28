#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "paudit/cli.hpp"

namespace fs = std::filesystem;
using paudit::cli::run;

namespace {

/// A small regression panel written to a temporary CSV.
fs::path write_demo() {
  const fs::path path = fs::temp_directory_path() / "paudit_cli_demo.csv";
  std::ofstream f(path);
  f << "t,g,y,pred,split\n";
  for (int i = 0; i < 200; ++i) {
    const double y = std::sin((i + 1) / 5.0);
    f << i << ',' << (i % 2) << ',' << y << ',' << std::sin(i / 5.0) << ',' << (i < 140 ? "train" : "test")
      << '\n';
  }
  return path;
}

std::vector<std::string> args_for(const fs::path& p, std::vector<std::string> extra) {
  std::vector<std::string> a = {p.string(), "--task", "regression", "--target", "y",       "--pred", "pred",
                                "--group",  "g",      "--time",     "t",        "--split", "split"};
  a.insert(a.end(), extra.begin(), extra.end());
  return a;
}

}  // namespace

TEST_CASE("the CLI audits a CSV and sets the exit code from --fail-on", "[cli]") {
  const fs::path p = write_demo();
  const auto r = run(args_for(p, {}));
  CHECK(r.err.empty());
  CHECK(r.out.starts_with("VERDICT: "));
  CHECK(r.out.find("[PASS] temporal split") != std::string::npos);
  CHECK(run(args_for(p, {"--fail-on", "never"})).exit_code == 0);
  CHECK(run(args_for(p, {"--format", "json"})).out.starts_with("{\n  \"task\": \"regression\","));
  CHECK(run(args_for(p, {"--format", "markdown"})).out.starts_with("**Verdict: "));
  fs::remove(p);
}

TEST_CASE("argument parsing follows argparse", "[cli]") {
  const fs::path p = write_demo();
  // Unique prefixes and --flag=value work; negative numbers are values, not options.
  CHECK(run(args_for(p, {"--fail=never", "--n-b", "50"})).exit_code == 0);
  CHECK(run(args_for(p, {"--seed", "-1"})).err.find("negative seeds") != std::string::npos);
  const auto ambiguous = run(args_for(p, {"--s", "1"}));
  CHECK(ambiguous.exit_code == 2);
  CHECK(ambiguous.err.find("ambiguous option: --s could match --split, --season, --seed") !=
        std::string::npos);
  CHECK(run(args_for(p, {"--horizon", "x"})).err.find("argument --horizon: invalid int value: 'x'") !=
        std::string::npos);
  CHECK(run(args_for(p, {"--bogus"})).err.find("unrecognized arguments: --bogus") != std::string::npos);
  CHECK(run({}).err.find("the following arguments are required: data") != std::string::npos);
  CHECK(run({"--help"}).out.starts_with("usage: predictive-audit"));
  fs::remove(p);
}

TEST_CASE("CLI errors and fallbacks", "[cli]") {
  const auto missing = run({"no_such_file.csv", "--target", "y"});
  CHECK(missing.exit_code == 2);
  CHECK(missing.err == "predictive-audit: error: data file not found: no_such_file.csv\n");

  const fs::path p = write_demo();
  CHECK(run(args_for(p, {"--target", "nope"})).err ==
        "predictive-audit: error: columns not found: ['nope']\n");
  CHECK(run({p.string()}).err == "predictive-audit: error: --target is required unless --wide is set\n");

  const fs::path parquet = fs::temp_directory_path() / "paudit_cli_demo.parquet";
  std::ofstream(parquet) << "x";
  paudit::cli::Options fallback;
  fallback.allow_fallback = true;
  const auto handed_back = run({parquet.string(), "--target", "y"}, fallback);
  CHECK(handed_back.fallback);
  CHECK(handed_back.out.empty());
  CHECK(run({parquet.string(), "--target", "y"}).exit_code == 2);  // standalone: an error
  fs::remove(parquet);
  fs::remove(p);
}

TEST_CASE("TOML config supplies defaults that flags override", "[cli]") {
  const fs::path p = write_demo();
  const fs::path cfg = fs::temp_directory_path() / "paudit_cli_demo.toml";
  std::ofstream(cfg) << "[audit]\nar-lags = 2\newma = true\nformat = \"json\"\nfail_on = \"never\"\n";
  const auto r = run(args_for(p, {"--config", cfg.string()}));
  CHECK(r.exit_code == 0);
  CHECK(r.out.find("\"linear AR(2)\"") != std::string::npos);
  CHECK(r.out.find("\"ewma (3)\"") != std::string::npos);
  CHECK(run(args_for(p, {"--config", cfg.string(), "--format", "text"})).out.starts_with("VERDICT"));
  CHECK(run(args_for(p, {"--config", "missing.toml"})).err ==
        "predictive-audit: error: config file not found: missing.toml\n");
  fs::remove(cfg);
  fs::remove(p);
}
