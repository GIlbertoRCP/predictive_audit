#include "paudit/version.hpp"

#include <thread>

#ifndef PAUDIT_VERSION
#error "PAUDIT_VERSION must be defined by the build system"
#endif
#ifndef PAUDIT_BUILD_TYPE
#define PAUDIT_BUILD_TYPE "unknown"
#endif

namespace paudit {
namespace {

#define PAUDIT_STR_IMPL(x) #x
#define PAUDIT_STR(x) PAUDIT_STR_IMPL(x)

constexpr std::string_view kCompiler =
#if defined(__clang__)
    "clang " __clang_version__;
#elif defined(__GNUC__)
    "gcc " __VERSION__;
#elif defined(_MSC_VER)
    "msvc " PAUDIT_STR(_MSC_FULL_VER);
#else
    "unknown";
#endif

constexpr long kCplusplus =
#if defined(_MSVC_LANG)
    _MSVC_LANG;
#else
    __cplusplus;
#endif

}  // namespace

std::string_view version() noexcept { return PAUDIT_VERSION; }

BuildInfo build_info() noexcept {
  return BuildInfo{
      .version = PAUDIT_VERSION,
      .compiler = kCompiler,
      .build_type = PAUDIT_BUILD_TYPE,
      .cplusplus = kCplusplus,
      .hardware_threads = std::thread::hardware_concurrency(),
  };
}

}  // namespace paudit
