// Standalone predictive-audit executable.
//
// Phase 0 scaffold: only --version / --build-info are implemented. The full CLI
// (same flags, reports and 0/1/2 exit codes as the Python CLI) lands in Phase 4;
// see docs/NATIVE_CORE_PLAN.md.

#include <cstdio>
#include <string_view>

#include "paudit/version.hpp"

namespace {

constexpr int kExitError = 2;

void print_usage(std::FILE* out) {
  std::fputs(
      "usage: predictive-audit [--version] [--build-info]\n"
      "\n"
      "Native predictive-audit engine (preview). Auditing data files is not\n"
      "implemented in the native CLI yet; use the Python CLI for now.\n",
      out);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2) {
    const std::string_view arg{argv[1]};
    if (arg == "--version") {
      const auto v = paudit::version();
      std::printf("predictive-audit %.*s (native)\n", static_cast<int>(v.size()), v.data());
      return 0;
    }
    if (arg == "--build-info") {
      const auto info = paudit::build_info();
      std::printf("version:          %.*s\n", static_cast<int>(info.version.size()), info.version.data());
      std::printf("compiler:         %.*s\n", static_cast<int>(info.compiler.size()), info.compiler.data());
      std::printf("build type:       %.*s\n", static_cast<int>(info.build_type.size()),
                  info.build_type.data());
      std::printf("c++ standard:     %ld\n", info.cplusplus);
      std::printf("hardware threads: %u\n", info.hardware_threads);
      return 0;
    }
    if (arg == "-h" || arg == "--help") {
      print_usage(stdout);
      return 0;
    }
  }
  print_usage(stderr);
  return kExitError;
}
