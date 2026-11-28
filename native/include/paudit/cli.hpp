#pragma once

// The predictive-audit command line, shared by the standalone executable and by the Python
// package's console script (through predictive_audit._native.cli_main).
//
// Flags, defaults, TOML config handling, --wide reshaping, messages and exit codes follow
// predictive_audit.cli: 0 = verdict below --fail-on, 1 = at or above it, 2 = usage or input
// error. On top of it, --threads N sets the engine's thread count.

#include <string>
#include <vector>

#include "paudit/csv.hpp"

namespace paudit::cli {

struct Options {
  /// Inputs the native engine does not model (Parquet/Feather, missing values in key columns,
  /// integers beyond int64, ...) return `fallback` instead of an error, so the Python package can
  /// run its pandas implementation. The standalone executable reports them as errors (exit 2).
  bool allow_fallback = false;
  /// pandas >= 3.0 span-3 EWMA weighting after gaps (see ewm_mean); the console script passes
  /// the installed pandas' behaviour, the standalone executable uses pandas 3's.
  bool ewm_com1_update = true;
  /// Which pandas read_csv to reproduce (the console script passes the installed one).
  io::Pandas pandas = io::Pandas::v31;
};

struct Result {
  int exit_code = 0;
  std::string out;  ///< text for stdout
  std::string err;  ///< text for stderr
  bool fallback = false;
  std::string fallback_reason;
};

/// Run `predictive-audit <args>` (args exclude the program name).
[[nodiscard]] Result run(const std::vector<std::string>& args, const Options& options = {});

}  // namespace paudit::cli
