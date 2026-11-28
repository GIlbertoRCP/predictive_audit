#include "paudit/report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pyformat.hpp"

namespace paudit {
namespace {

using detail::py_fixed;
using detail::py_len;

std::string_view task_name(Task t) { return t == Task::classification ? "classification" : "regression"; }

// report._fmt
std::string fmt_value(double x) { return std::isfinite(x) ? py_fixed(x, 4) : "n/a"; }

struct Rows {
  std::vector<std::string> names;
  std::vector<std::vector<std::string>> rows;
};

// report._rows
Rows table_rows(const AuditReport& r) {
  Rows out;
  if (r.table.empty()) return out;
  for (const auto& [metric, value] : r.table.front().second) out.names.emplace_back(metric_name(metric));
  for (const auto& [model, values] : r.table) {
    std::string label = model;
    if (r.model && model == *r.model) {
      label += "  <- model";
    } else if (model == r.best_baseline) {
      label += "  <- best baseline";
    }
    std::vector<std::string> row{label};
    for (const auto& [metric, value] : values) row.push_back(fmt_value(value));
    out.rows.push_back(std::move(row));
  }
  return out;
}

// report._header
std::vector<std::string> header_lines(const AuditReport& r) {
  std::vector<std::string> lines;
  lines.push_back("target: " + r.target + "   task: " + std::string(task_name(r.task)) + "   horizon: " +
                  std::to_string(r.horizon) + "   metric: " + std::string(metric_name(r.metric)) +
                  "   rows scored: " + std::to_string(r.n_eval));
  if (r.season && *r.season != 0) lines.push_back("seasonal period: " + std::to_string(*r.season) + " steps");
  if (r.prevalence) lines.push_back("positive rate: " + py_fixed(*r.prevalence, 3));
  if (r.threshold && *r.threshold != 0.5)
    lines.push_back("classification threshold: " + py_fixed(*r.threshold, 4));
  if (r.mase && std::isfinite(*r.mase)) lines.push_back("MASE (vs persistence): " + py_fixed(*r.mase, 4));
  return lines;
}

std::string join(const std::vector<std::string>& parts, std::string_view sep) {
  std::string out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i != 0) out += sep;
    out += parts[i];
  }
  return out;
}

// --- JSON ---------------------------------------------------------------------------------------

struct Json {
  enum class Type : std::uint8_t { null, real, integer, string, array, object } type = Type::null;
  double real = 0.0;
  std::int64_t integer = 0;
  std::string str;
  std::vector<Json> items;
  std::vector<std::pair<std::string, Json>> fields;

  static Json number(double v) {
    Json j;
    j.type = std::isfinite(v) ? Type::real : Type::null;  // RFC 8259 has no NaN/Inf
    j.real = v;
    return j;
  }
  static Json whole(std::int64_t v) {
    Json j;
    j.type = Type::integer;
    j.integer = v;
    return j;
  }
  static Json text(std::string v) {
    Json j;
    j.type = Type::string;
    j.str = std::move(v);
    return j;
  }
  static Json array() {
    Json j;
    j.type = Type::array;
    return j;
  }
  static Json object() {
    Json j;
    j.type = Type::object;
    return j;
  }
  Json& add(std::string key, Json value) {
    fields.emplace_back(std::move(key), std::move(value));
    return *this;
  }
};

void append_u16(std::string& out, std::uint32_t unit) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "\\u%04x", unit);
  out += buf;
}

// json.dumps(ensure_ascii=True): everything outside ' '..'~' is escaped, astral code points as
// surrogate pairs.
void write_string(std::string& out, std::string_view s) {
  out += '"';
  std::size_t i = 0;
  while (i < s.size()) {
    const auto c = static_cast<unsigned char>(s[i]);
    std::uint32_t cp = c;
    std::size_t len = 1;
    if (c >= 0xF0) {
      len = 4;
      cp = c & 0x07u;
    } else if (c >= 0xE0) {
      len = 3;
      cp = c & 0x0Fu;
    } else if (c >= 0xC0) {
      len = 2;
      cp = c & 0x1Fu;
    }
    for (std::size_t k = 1; k < len && i + k < s.size(); ++k) {
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
    }
    i += len;
    switch (cp) {
      case '"':
        out += "\\\"";
        continue;
      case '\\':
        out += "\\\\";
        continue;
      case '\n':
        out += "\\n";
        continue;
      case '\r':
        out += "\\r";
        continue;
      case '\t':
        out += "\\t";
        continue;
      case '\b':
        out += "\\b";
        continue;
      case '\f':
        out += "\\f";
        continue;
      default:
        break;
    }
    if (cp >= 0x20 && cp <= 0x7E) {
      out += static_cast<char>(cp);
    } else if (cp > 0xFFFF) {
      const std::uint32_t v = cp - 0x10000;
      append_u16(out, 0xD800 + (v >> 10));
      append_u16(out, 0xDC00 + (v & 0x3FF));
    } else {
      append_u16(out, cp);
    }
  }
  out += '"';
}

void dump(std::string& out, const Json& j, std::size_t depth) {
  const auto indent = [&](std::size_t d) { out.append(2 * d, ' '); };
  switch (j.type) {
    case Json::Type::null:
      out += "null";
      return;
    case Json::Type::real:
      out += detail::py_repr(j.real);
      return;
    case Json::Type::integer:
      out += std::to_string(j.integer);
      return;
    case Json::Type::string:
      write_string(out, j.str);
      return;
    case Json::Type::array:
      if (j.items.empty()) {
        out += "[]";
        return;
      }
      out += "[\n";
      for (std::size_t i = 0; i < j.items.size(); ++i) {
        indent(depth + 1);
        dump(out, j.items[i], depth + 1);
        out += i + 1 < j.items.size() ? ",\n" : "\n";
      }
      indent(depth);
      out += ']';
      return;
    case Json::Type::object:
      if (j.fields.empty()) {
        out += "{}";
        return;
      }
      out += "{\n";
      for (std::size_t i = 0; i < j.fields.size(); ++i) {
        indent(depth + 1);
        write_string(out, j.fields[i].first);
        out += ": ";
        dump(out, j.fields[i].second, depth + 1);
        out += i + 1 < j.fields.size() ? ",\n" : "\n";
      }
      indent(depth);
      out += '}';
      return;
  }
}

Json detail_json(const DetailValue& v) {
  return std::visit(
      [](const auto& x) -> Json {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return Json{};
        } else if constexpr (std::is_same_v<T, double>) {
          return Json::number(x);
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
          return Json::whole(x);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return Json::text(x);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          Json a = Json::array();
          for (const double d : x) a.items.push_back(Json::number(d));
          return a;
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          Json a = Json::array();
          for (const auto& s : x) a.items.push_back(Json::text(s));
          return a;
        } else {
          Json o = Json::object();
          for (const auto& [k, d] : x) o.add(k, Json::number(d));
          return o;
        }
      },
      v);
}

template <class T, class F>
Json optional_json(const std::optional<T>& v, F make) {
  return v ? make(*v) : Json{};
}

}  // namespace

std::string to_text(const AuditReport& r) {
  const Rows rows = table_rows(r);
  std::vector<std::string> head{"model"};
  head.insert(head.end(), rows.names.begin(), rows.names.end());
  std::vector<std::size_t> widths(head.size(), 0);
  for (std::size_t i = 0; i < head.size(); ++i) {
    widths[i] = py_len(head[i]);
    for (const auto& row : rows.rows) widths[i] = std::max(widths[i], py_len(row[i]));
  }
  const auto line = [&](const std::vector<std::string>& cells) {
    std::string out;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      if (i != 0) out += "  ";
      out += cells[i];
      const std::size_t len = py_len(cells[i]);
      if (len < widths[i]) out.append(widths[i] - len, ' ');  // str.ljust pads every column
    }
    return out;
  };

  std::vector<std::string> out{"VERDICT: " + std::string(status_name(r.verdict()))};
  for (auto& h : header_lines(r)) out.push_back(std::move(h));
  out.emplace_back();
  out.push_back(line(head));
  for (const auto& row : rows.rows) out.push_back(line(row));
  out.emplace_back();
  for (const auto& c : r.checks) {
    out.push_back("[" + std::string(status_name(c.status)) + "] " + c.name + ": " + c.summary);
  }
  return join(out, "\n");
}

std::string to_markdown(const AuditReport& r) {
  const Rows rows = table_rows(r);
  std::vector<std::string> out{"**Verdict: " + std::string(status_name(r.verdict())) + "**", ""};
  for (const auto& h : header_lines(r)) out.push_back("- " + h);
  out.emplace_back();
  out.push_back("| model | " + join(rows.names, " | ") + " |");
  std::string rule;
  for (std::size_t i = 0; i < rows.names.size() + 1; ++i) rule += "|---";
  out.push_back(rule + "|");
  for (const auto& row : rows.rows) out.push_back("| " + join(row, " | ") + " |");
  out.emplace_back();
  out.emplace_back("| check | status | finding |");
  out.emplace_back("|---|---|---|");
  for (const auto& c : r.checks) {
    out.push_back("| " + c.name + " | " + std::string(status_name(c.status)) + " | " + c.summary + " |");
  }
  return join(out, "\n");
}

std::string to_json(const AuditReport& r) {
  Json root = Json::object();
  root.add("task", Json::text(std::string(task_name(r.task))));
  root.add("target", Json::text(r.target));
  root.add("horizon", Json::whole(r.horizon));
  root.add("metric", Json::text(std::string(metric_name(r.metric))));
  root.add("model", optional_json(r.model, [](const std::string& s) { return Json::text(s); }));
  root.add("season", optional_json(r.season, [](std::int64_t s) { return Json::whole(s); }));
  root.add("n_eval", Json::whole(static_cast<std::int64_t>(r.n_eval)));
  root.add("prevalence", optional_json(r.prevalence, [](double v) { return Json::number(v); }));
  root.add("best_baseline", Json::text(r.best_baseline));
  Json table = Json::object();
  for (const auto& [model, values] : r.table) {
    Json row = Json::object();
    for (const auto& [metric, v] : values) row.add(std::string(metric_name(metric)), Json::number(v));
    table.add(model, std::move(row));
  }
  root.add("table", std::move(table));
  Json checks = Json::array();
  for (const auto& c : r.checks) {
    Json check = Json::object();
    check.add("name", Json::text(c.name));
    check.add("status", Json::text(std::string(status_name(c.status))));
    check.add("summary", Json::text(c.summary));
    Json details = Json::object();
    for (const auto& [k, v] : c.details) details.add(k, detail_json(v));
    check.add("details", std::move(details));
    checks.items.push_back(std::move(check));
  }
  root.add("checks", std::move(checks));
  root.add("schema_version", Json::text(kSchemaVersion));
  root.add("threshold", optional_json(r.threshold, [](double v) { return Json::number(v); }));
  root.add("mase", optional_json(r.mase, [](double v) { return Json::number(v); }));
  root.add("verdict", Json::text(std::string(status_name(r.verdict()))));
  std::string out;
  dump(out, root, 0);
  return out;
}

}  // namespace paudit
