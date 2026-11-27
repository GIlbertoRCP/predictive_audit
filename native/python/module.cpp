// nanobind extension: predictive_audit._native
//
// Phase 0 exposes build metadata only. Kernels are added phase by phase behind
// the same Python signatures; see docs/NATIVE_CORE_PLAN.md.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

#include "paudit/version.hpp"

namespace nb = nanobind;

NB_MODULE(_native, m) {
  m.doc() = "Native C++ engine for predictive-audit.";

  m.def(
      "version", [] { return std::string{paudit::version()}; },
      "Version of the native engine (matches the Python package version).");

  m.def(
      "build_info",
      [] {
        const auto info = paudit::build_info();
        nb::dict d;
        d["version"] = std::string{info.version};
        d["compiler"] = std::string{info.compiler};
        d["build_type"] = std::string{info.build_type};
        d["cplusplus"] = info.cplusplus;
        d["hardware_threads"] = info.hardware_threads;
        return d;
      },
      "Compiler, C++ standard, build type and hardware thread count of this build.");
}
