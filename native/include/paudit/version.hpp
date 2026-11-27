#pragma once

#include <string_view>

namespace paudit {

/// Library version, kept in sync with pyproject.toml at configure time.
[[nodiscard]] std::string_view version() noexcept;

/// Static facts about how this copy of the engine was built.
struct BuildInfo {
  std::string_view version;
  std::string_view compiler;
  std::string_view build_type;
  long cplusplus;             ///< value of __cplusplus (or _MSVC_LANG)
  unsigned hardware_threads;  ///< std::thread::hardware_concurrency(), 0 if unknown
};

[[nodiscard]] BuildInfo build_info() noexcept;

}  // namespace paudit
