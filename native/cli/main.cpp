// Standalone predictive-audit executable: the full CLI with no Python on the machine.
// Same flags, reports and 0/1/2 exit codes as the Python package's console script (both call
// paudit::cli::run). Inputs only the Python package can read (Parquet, Feather, ...) exit 2 here.

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "paudit/cli.hpp"
#include "paudit/version.hpp"

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() == 1 && args[0] == "--build-info") {
    const auto info = paudit::build_info();
    std::printf("version:          %.*s\n", static_cast<int>(info.version.size()), info.version.data());
    std::printf("compiler:         %.*s\n", static_cast<int>(info.compiler.size()), info.compiler.data());
    std::printf("build type:       %.*s\n", static_cast<int>(info.build_type.size()), info.build_type.data());
    std::printf("c++ standard:     %ld\n", info.cplusplus);
    std::printf("hardware threads: %u\n", info.hardware_threads);
    return 0;
  }
  const paudit::cli::Result result = paudit::cli::run(args);
  std::fwrite(result.out.data(), 1, result.out.size(), stdout);
  std::fwrite(result.err.data(), 1, result.err.size(), stderr);
  return result.exit_code;
}
