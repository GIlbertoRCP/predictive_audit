#pragma once

// Report renderers, byte-identical to predictive_audit.report: to_text, to_markdown, and to_json
// (strict RFC 8259 in json.dumps(indent=2) layout, ASCII-escaped, NaN/Inf as null, floats in
// Python repr form).

#include <string>

#include "paudit/audit.hpp"

namespace paudit {

[[nodiscard]] std::string to_text(const AuditReport& report);
[[nodiscard]] std::string to_markdown(const AuditReport& report);
[[nodiscard]] std::string to_json(const AuditReport& report);

/// The report schema version written to JSON (predictive_audit.SCHEMA_VERSION).
inline constexpr const char* kSchemaVersion = "1.0";

}  // namespace paudit
