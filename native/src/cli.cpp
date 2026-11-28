#include "paudit/cli.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// toml++ is included as a system header (see CMakeLists.txt), so its own warnings are not ours.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <toml++/toml.hpp>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "paudit/audit.hpp"
#include "paudit/csv.hpp"
#include "paudit/metrics.hpp"
#include "paudit/parallel.hpp"
#include "paudit/report.hpp"
#include "paudit/version.hpp"
#include "pyformat.hpp"

namespace paudit::cli {
namespace {

namespace fs = std::filesystem;
using detail::py_repr_str;
using io::Column;
using io::ColumnKind;
using io::Table;
using io::Unsupported;

constexpr int kExitError = 2;

/// argparse-style error: usage line, then "predictive-audit: error: ...".
struct UsageError {
  std::string message;
};
/// predictive_audit.cli's own error: "predictive-audit: error: ...".
struct CliError {
  std::string message;
};

// --- Python int() / float() -------------------------------------------------------------------

constexpr bool py_space(char c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r' ||
         (c >= '\x1c' && c <= '\x1f');
}
constexpr bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }
constexpr char lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

std::string_view py_strip(std::string_view s) noexcept {
  while (!s.empty() && py_space(s.front())) s.remove_prefix(1);
  while (!s.empty() && py_space(s.back())) s.remove_suffix(1);
  return s;
}

std::string lowered(std::string_view s) {
  std::string out(s);
  for (auto& c : out) c = lower(c);
  return out;
}

/// Digits with single underscores between them (Python's numeric literal rule), or nullopt.
/// `pos` advances past the digits.
std::optional<std::string> digit_part(std::string_view s, std::size_t& pos) {
  std::string out;
  if (pos >= s.size() || !is_digit(s[pos])) return std::nullopt;
  while (pos < s.size()) {
    if (is_digit(s[pos])) {
      out += s[pos++];
    } else if (s[pos] == '_' && pos + 1 < s.size() && is_digit(s[pos + 1])) {
      ++pos;
    } else {
      break;
    }
  }
  return out;
}

enum class IntStatus : std::uint8_t { ok, invalid, overflow };

/// int(text) for base-10 text. Values outside int64 report `overflow` (Python would accept them).
IntStatus py_int(std::string_view text, std::int64_t& out) {
  const std::string_view s = py_strip(text);
  std::size_t pos = 0;
  bool negative = false;
  if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) negative = s[pos++] == '-';
  const auto digits = digit_part(s, pos);
  if (!digits || pos != s.size()) return IntStatus::invalid;
  std::uint64_t acc = 0;
  const std::uint64_t limit = negative ? std::uint64_t{1} << 63 : (std::uint64_t{1} << 63) - 1;
  for (const char c : *digits) {
    const auto d = static_cast<std::uint64_t>(c - '0');
    if (acc > (limit - d) / 10) return IntStatus::overflow;
    acc = acc * 10 + d;
  }
  out = negative ? (acc == (std::uint64_t{1} << 63) ? std::numeric_limits<std::int64_t>::min()
                                                    : -static_cast<std::int64_t>(acc))
                 : static_cast<std::int64_t>(acc);
  return IntStatus::ok;
}

/// float(text): decimal or scientific notation with optional underscores, or inf/infinity/nan.
std::optional<double> py_float(std::string_view text) {
  const std::string_view s = py_strip(text);
  std::size_t pos = 0;
  std::string clean;
  if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) clean += s[pos++];
  const std::string rest = lowered(s.substr(pos));
  if (rest == "inf" || rest == "infinity") return clean == "-" ? -HUGE_VAL : HUGE_VAL;
  if (rest == "nan") return std::numeric_limits<double>::quiet_NaN();
  bool any_digits = false;
  if (const auto int_part = digit_part(s, pos)) {
    clean += *int_part;
    any_digits = true;
  }
  if (pos < s.size() && s[pos] == '.') {
    clean += s[pos++];
    if (const auto frac = digit_part(s, pos)) {
      clean += *frac;
      any_digits = true;
    }
  }
  if (!any_digits) return std::nullopt;
  if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
    clean += 'e';
    ++pos;
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) clean += s[pos++];
    const auto exp = digit_part(s, pos);
    if (!exp) return std::nullopt;
    clean += *exp;
  }
  if (pos != s.size()) return std::nullopt;
  return std::strtod(clean.c_str(), nullptr);  // correctly rounded, like Python
}

// --- the argument parser -------------------------------------------------------------------------

enum class Kind : std::uint8_t { value, flag, help, version };
enum class Type : std::uint8_t { text, integer, real, season, threshold, block, choice };

struct Spec {
  std::string_view flag;
  std::string_view dest;
  Kind kind;
  Type type = Type::text;
  std::span<const std::string_view> choices = {};
};

constexpr std::array<std::string_view, 2> kTasks = {"classification", "regression"};
constexpr std::array<std::string_view, 3> kFormats = {"text", "markdown", "json"};
constexpr std::array<std::string_view, 3> kFailOn = {"fail", "warn", "never"};

// In predictive_audit.cli.build_parser order (argparse lists ambiguous matches in this order).
constexpr std::array<Spec, 28> kSpecs = {{
    {"--help", "help", Kind::help},
    {"--config", "config", Kind::value},
    {"--target", "target", Kind::value},
    {"--task", "task", Kind::value, Type::choice, kTasks},
    {"--horizon", "horizon", Kind::value, Type::integer},
    {"--pred", "pred", Kind::value},
    {"--model-name", "model_name", Kind::value},
    {"--group", "group", Kind::value},
    {"--time", "time", Kind::value},
    {"--split", "split", Kind::value},
    {"--features", "features", Kind::value},
    {"--wide", "wide", Kind::flag},
    {"--wide-keep", "wide_keep", Kind::value},
    {"--season", "season", Kind::value, Type::season},
    {"--window", "window", Kind::value, Type::integer},
    {"--ewma", "ewma", Kind::flag},
    {"--ar-lags", "ar_lags", Kind::value, Type::integer},
    {"--metric", "metric", Kind::value},
    {"--reported", "reported", Kind::value, Type::real},
    {"--threshold", "threshold", Kind::value, Type::threshold},
    {"--n-boot", "n_boot", Kind::value, Type::integer},
    {"--block", "block", Kind::value, Type::block},
    {"--seed", "seed", Kind::value, Type::integer},
    {"--format", "format", Kind::value, Type::choice, kFormats},
    {"--out", "out", Kind::value},
    {"--fail-on", "fail_on", Kind::value, Type::choice, kFailOn},
    {"--threads", "threads", Kind::value, Type::integer},
    {"--version", "version", Kind::version},
}};

constexpr std::string_view kUsage =
    "usage: predictive-audit [-h] [--config CONFIG] [--target TARGET]\n"
    "                        [--task {classification,regression}]\n"
    "                        [--horizon HORIZON] [--pred PRED]\n"
    "                        [--model-name MODEL_NAME] [--group GROUP]\n"
    "                        [--time TIME] [--split SPLIT] [--features FEATURES]\n"
    "                        [--wide] [--wide-keep WIDE_KEEP] [--season SEASON]\n"
    "                        [--window WINDOW] [--ewma] [--ar-lags AR_LAGS]\n"
    "                        [--metric METRIC] [--reported REPORTED]\n"
    "                        [--threshold THRESHOLD] [--n-boot N_BOOT]\n"
    "                        [--block BLOCK] [--seed SEED]\n"
    "                        [--format {text,markdown,json}] [--out OUT]\n"
    "                        [--fail-on {fail,warn,never}] [--threads THREADS]\n"
    "                        [--version]\n"
    "                        data\n";

constexpr std::string_view kHelpBody =
    "\n"
    "Check whether a time-series forecasting result beats trivial baselines.\n"
    "\n"
    "positional arguments:\n"
    "  data                  CSV, TSV, Parquet, or Feather table\n"
    "\n"
    "options:\n"
    "  -h, --help            show this help message and exit\n"
    "  --config CONFIG       TOML config file with default audit options\n"
    "  --target TARGET       column holding y[t + horizon] (auto-set to 'value' with\n"
    "                        --wide)\n"
    "  --task {classification,regression}\n"
    "  --horizon HORIZON     steps ahead the target is (default 1)\n"
    "  --pred PRED           column with the model's predictions/scores\n"
    "  --model-name MODEL_NAME\n"
    "                        display name for the model\n"
    "  --group GROUP         comma-separated columns identifying each series\n"
    "  --time TIME           time column (otherwise file order within each series)\n"
    "  --split SPLIT         column labelling rows test / train / other; only test\n"
    "                        rows are scored, other rows are history\n"
    "  --features FEATURES   comma-separated feature columns to scan for future\n"
    "                        leakage\n"
    "  --wide                reshape wide matrix (one column per series) before\n"
    "                        auditing\n"
    "  --wide-keep WIDE_KEEP\n"
    "                        comma-separated non-series columns to preserve with\n"
    "                        --wide\n"
    "  --season SEASON       season length for a seasonal-naive baseline, or 'auto'\n"
    "  --window WINDOW       moving-average window (default max(3, horizon))\n"
    "  --ewma                include an EWMA baseline\n"
    "  --ar-lags AR_LAGS     include a causal Ridge AR(p) baseline fit on non-test\n"
    "                        rows\n"
    "  --metric METRIC       headline metric (default f1 / mae)\n"
    "  --reported REPORTED   a published score on --metric to check\n"
    "  --threshold THRESHOLD\n"
    "                        classification score threshold (float or 'auto' tuned\n"
    "                        on non-test rows)\n"
    "  --n-boot N_BOOT\n"
    "  --block BLOCK         bootstrap block length (int, 'auto' Politis-White, or\n"
    "                        default)\n"
    "  --seed SEED\n"
    "  --format {text,markdown,json}\n"
    "  --out OUT             write the report here instead of stdout\n"
    "  --fail-on {fail,warn,never}\n"
    "                        exit non-zero at this verdict or worse (for CI)\n"
    "  --threads THREADS     engine threads (default: PREDICTIVE_AUDIT_THREADS or\n"
    "                        all cores)\n"
    "  --version             show program's version number and exit\n";

std::string type_label(Type t) {
  switch (t) {
    case Type::integer:
      return "int";
    case Type::real:
      return "float";
    case Type::season:
      return "_season";
    case Type::threshold:
      return "_threshold";
    case Type::block:
      return "_block";
    case Type::text:
    case Type::choice:
      break;
  }
  return "str";
}

/// argparse's type conversion check for a value given on the command line.
void check_value(const Spec& spec, const std::string& v, bool check_choices) {
  const auto invalid = [&] {
    return UsageError{"argument " + std::string(spec.flag) + ": invalid " + type_label(spec.type) +
                      " value: " + py_repr_str(v)};
  };
  std::int64_t i = 0;
  switch (spec.type) {
    case Type::integer:
      if (py_int(v, i) == IntStatus::invalid) throw invalid();
      return;
    case Type::real:
      if (!py_float(v)) throw invalid();
      return;
    case Type::season:
    case Type::block:
      if (v != "auto" && py_int(v, i) == IntStatus::invalid) throw invalid();
      return;
    case Type::threshold:
      if (v != "auto" && !py_float(v)) throw invalid();
      return;
    case Type::choice:
      if (check_choices && std::find(spec.choices.begin(), spec.choices.end(), v) == spec.choices.end()) {
        std::string opts;  // Python >= 3.12.8 wording (no quotes around the choices)
        for (std::size_t k = 0; k < spec.choices.size(); ++k)
          opts += (k ? ", " : "") + std::string(spec.choices[k]);
        throw UsageError{"argument " + std::string(spec.flag) + ": invalid choice: " + py_repr_str(v) +
                         " (choose from " + opts + ")"};
      }
      return;
    case Type::text:
      return;
  }
}

bool negative_number(std::string_view s) {
  // argparse: '^-\d+$|^-\d*\.\d+$'
  if (s.size() < 2 || s[0] != '-') return false;
  const std::string_view body = s.substr(1);
  const std::size_t dot = body.find('.');
  const auto all_digits = [](std::string_view d) { return std::all_of(d.begin(), d.end(), is_digit); };
  if (dot == std::string_view::npos) return all_digits(body);
  return all_digits(body.substr(0, dot)) && dot + 1 < body.size() && all_digits(body.substr(dot + 1));
}

bool looks_like_option(std::string_view s) {
  return s.size() > 1 && s[0] == '-' && !negative_number(s) && s.find(' ') == std::string_view::npos;
}

struct Parsed {
  std::optional<std::string> data;
  std::map<std::string, std::string, std::less<>> values;
  std::set<std::string, std::less<>> flags;
  std::vector<std::string> unrecognized;
  bool help = false;
  bool version = false;
};

const Spec* match_option(std::string_view name) {
  for (const auto& s : kSpecs) {
    if (s.flag == name) return &s;
  }
  std::vector<const Spec*> hits;
  for (const auto& s : kSpecs) {
    if (s.flag.starts_with(name)) hits.push_back(&s);
  }
  if (hits.size() > 1) {
    std::string list;
    for (std::size_t k = 0; k < hits.size(); ++k) list += (k ? ", " : "") + std::string(hits[k]->flag);
    throw UsageError{"ambiguous option: " + std::string(name) + " could match " + list};
  }
  return hits.empty() ? nullptr : hits.front();
}

Parsed parse_args(const std::vector<std::string>& args) {
  Parsed p;
  bool only_positional = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    const auto positional = [&] {
      if (!p.data) {
        p.data = a;
      } else {
        p.unrecognized.push_back(a);
      }
    };
    if (only_positional) {
      positional();
      continue;
    }
    if (a == "--") {
      only_positional = true;
      continue;
    }
    if (a == "-h") {
      p.help = true;
      return p;
    }
    if (a.starts_with("--") && a.size() > 2) {
      const std::size_t eq = a.find('=');
      const std::string name = a.substr(0, eq);
      const Spec* spec = match_option(name);
      if (spec == nullptr) {
        p.unrecognized.push_back(a);
        continue;
      }
      if (spec->kind != Kind::value && eq != std::string::npos) {
        throw UsageError{"argument " + std::string(spec->flag) + ": ignored explicit argument " +
                         py_repr_str(a.substr(eq + 1))};
      }
      switch (spec->kind) {
        case Kind::help:
          p.help = true;
          return p;
        case Kind::version:
          p.version = true;
          return p;
        case Kind::flag:
          p.flags.insert(std::string(spec->dest));
          continue;
        case Kind::value:
          break;
      }
      std::string v;
      if (eq != std::string::npos) {
        v = a.substr(eq + 1);
      } else if (i + 1 < args.size() && !looks_like_option(args[i + 1])) {
        v = args[++i];
      } else {
        throw UsageError{"argument " + std::string(spec->flag) + ": expected one argument"};
      }
      check_value(*spec, v, true);
      p.values[std::string(spec->dest)] = v;
      continue;
    }
    if (looks_like_option(a)) {
      p.unrecognized.push_back(a);
      continue;
    }
    positional();
  }
  return p;
}

// --- TOML defaults (predictive_audit.cli._load_toml_defaults) ------------------------------------

/// A config value: text, an integer, a float, a bool, or a type the CLI does not accept.
struct Unusable {};
using Default = std::variant<std::string, std::int64_t, double, bool, Unusable>;

/// str(Path(text)): repeated and trailing separators and "." components removed.
std::string py_path(const std::string& text) {
#ifdef _WIN32
  return text;
#else
  if (text.empty()) return ".";
  std::string prefix;
  if (text.starts_with("//") && !text.starts_with("///")) {
    prefix = "//";
  } else if (text.starts_with("/")) {
    prefix = "/";
  }
  std::vector<std::string> parts;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, '/')) {
    if (!part.empty() && part != ".") parts.push_back(part);
  }
  std::string out = prefix;
  for (std::size_t k = 0; k < parts.size(); ++k) out += (k ? "/" : "") + parts[k];
  return out.empty() ? "." : out;
#endif
}

std::string toml_str(const toml::node& n) {  // str(x) of a list item
  if (const auto* s = n.as_string()) return std::string(s->get());
  if (const auto* i = n.as_integer()) return std::to_string(i->get());
  if (const auto* f = n.as_floating_point()) return detail::py_repr(f->get());
  if (const auto* b = n.as_boolean()) return b->get() ? "True" : "False";
  throw Unsupported("config lists may only hold strings, numbers and booleans");
}

std::map<std::string, Default, std::less<>> load_toml(const std::string& path_text) {
  const fs::path path(path_text);
  std::error_code ec;
  if (!fs::exists(path, ec)) throw CliError{"config file not found: " + py_path(path_text)};
  toml::table root;
  try {
    root = toml::parse_file(path.string());
  } catch (const toml::parse_error& e) {
    std::ostringstream msg;
    msg << e.description() << " (at line " << e.source().begin.line << ", column " << e.source().begin.column
        << ")";
    throw CliError{msg.str()};
  }
  const toml::table* section = &root;
  if (const toml::node* audit = root.get("audit")) {
    section = audit->as_table();
    if (section == nullptr) throw CliError{"config file must contain a top-level mapping or [audit] table"};
  }
  std::map<std::string, Default, std::less<>> out;
  for (const auto& [key, node] : *section) {
    std::string k(key.str());
    std::replace(k.begin(), k.end(), '-', '_');
    if (const auto* s = node.as_string()) {
      out[k] = std::string(s->get());
    } else if (const auto* i = node.as_integer()) {
      out[k] = i->get();
    } else if (const auto* f = node.as_floating_point()) {
      out[k] = f->get();
    } else if (const auto* b = node.as_boolean()) {
      out[k] = b->get();
    } else if (const auto* arr = node.as_array()) {
      std::string joined;
      bool ok = true;
      for (std::size_t j = 0; j < arr->size(); ++j) {
        const toml::node* item = arr->get(j);
        if (item == nullptr || item->is_array() || item->is_table()) {
          ok = false;
          break;
        }
        joined += (j ? "," : "") + toml_str(*item);
      }
      if (ok) {
        out[k] = joined;
      } else {
        out[k] = Unusable{};
      }
    } else {
      out[k] = Unusable{};
    }
  }
  return out;
}

// --- resolved settings -------------------------------------------------------------------------

struct Settings {
  std::string data;
  std::optional<std::string> target, pred, model_name, group, time, split, features, wide_keep, metric, out;
  std::string task = "classification";
  std::string format = "text";
  std::string fail_on = "fail";
  std::int64_t horizon = 1;
  std::int64_t n_boot = 500;
  std::int64_t seed = 0;
  std::optional<std::int64_t> window, ar_lags, threads;
  bool wide = false;
  bool ewma = false;
  AuditConfig::SeasonMode season_mode = AuditConfig::SeasonMode::none;
  std::int64_t season = 0;
  bool threshold_auto = false;
  double threshold = 0.5;
  AuditConfig::BlockMode block_mode = AuditConfig::BlockMode::automatic_default;
  std::int64_t block = 0;
  std::optional<double> reported;
  bool reported_is_int = false;  ///< a TOML integer: Python keeps it an int in the JSON details
};

const Spec& spec_for(std::string_view dest) {
  for (const auto& s : kSpecs) {
    if (s.dest == dest) return s;
  }
  throw std::logic_error("unknown option");
}

class Resolver {
 public:
  Resolver(const Parsed& p, const std::map<std::string, Default, std::less<>>& defaults)
      : parsed_(p), defaults_(defaults) {}

  /// The text a string-valued source supplies (the CLI, or a TOML string), else the TOML value.
  [[nodiscard]] std::optional<Default> source(std::string_view dest) const {
    if (const auto it = parsed_.values.find(dest); it != parsed_.values.end()) return Default{it->second};
    if (const auto it = defaults_.find(dest); it != defaults_.end()) {
      if (std::holds_alternative<std::string>(it->second)) {
        check_value(spec_for(dest), std::get<std::string>(it->second),
                    false);  // argparse converts string defaults
      }
      return it->second;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<std::string> text(std::string_view dest) const {
    const auto v = source(dest);
    if (!v) return std::nullopt;
    if (const auto* s = std::get_if<std::string>(&*v)) return *s;
    throw Unsupported("config value for '" + std::string(dest) + "' is not a string");
  }

  [[nodiscard]] std::optional<std::int64_t> integer(std::string_view dest) const {
    const auto v = source(dest);
    if (!v) return std::nullopt;
    if (const auto* i = std::get_if<std::int64_t>(&*v)) return *i;
    if (const auto* s = std::get_if<std::string>(&*v)) {
      std::int64_t out = 0;
      if (py_int(*s, out) != IntStatus::ok)
        throw Unsupported("--" + std::string(dest) + " is outside the int64 range");
      return out;
    }
    throw Unsupported("config value for '" + std::string(dest) + "' is not an integer");
  }

  [[nodiscard]] bool flag(std::string_view dest) const {
    if (parsed_.flags.contains(dest)) return true;
    const auto it = defaults_.find(dest);
    if (it == defaults_.end()) return false;
    return std::visit(
        [](const auto& x) -> bool {  // Python truthiness of the raw default
          using T = std::decay_t<decltype(x)>;
          if constexpr (std::is_same_v<T, std::string>) {
            return !x.empty();
          } else if constexpr (std::is_same_v<T, Unusable>) {
            throw Unsupported("unsupported config value for a flag");
          } else {
            return x != T{};
          }
        },
        it->second);
  }

 private:
  const Parsed& parsed_;
  const std::map<std::string, Default, std::less<>>& defaults_;
};

Settings resolve(const Parsed& p, const std::map<std::string, Default, std::less<>>& defaults) {
  const Resolver r(p, defaults);
  Settings s;
  s.data = *p.data;
  s.target = r.text("target");
  s.pred = r.text("pred");
  s.model_name = r.text("model_name");
  s.group = r.text("group");
  s.time = r.text("time");
  s.split = r.text("split");
  s.features = r.text("features");
  s.wide_keep = r.text("wide_keep");
  s.metric = r.text("metric");
  s.out = r.text("out");
  if (const auto v = r.text("task")) s.task = *v;
  if (const auto v = r.text("format")) s.format = *v;
  if (const auto v = r.text("fail_on")) s.fail_on = *v;
  if (const auto v = r.integer("horizon")) s.horizon = *v;
  if (const auto v = r.integer("n_boot")) s.n_boot = *v;
  if (const auto v = r.integer("seed")) s.seed = *v;
  s.window = r.integer("window");
  s.ar_lags = r.integer("ar_lags");
  s.threads = r.integer("threads");
  s.wide = r.flag("wide");
  s.ewma = r.flag("ewma");

  if (const auto v = r.source("season")) {
    if (const auto* text = std::get_if<std::string>(&*v); text && *text == "auto") {
      s.season_mode = AuditConfig::SeasonMode::automatic;
    } else {
      const auto m = r.integer("season");
      s.season_mode = AuditConfig::SeasonMode::fixed;
      s.season = *m;
    }
  }
  if (const auto v = r.source("threshold")) {
    if (const auto* text = std::get_if<std::string>(&*v)) {
      if (*text == "auto") {
        s.threshold_auto = true;
      } else {
        s.threshold = *py_float(*text);
      }
    } else if (const auto* d = std::get_if<double>(&*v)) {
      s.threshold = *d;
    } else if (const auto* i = std::get_if<std::int64_t>(&*v)) {
      s.threshold = static_cast<double>(*i);
    } else {
      throw Unsupported("config value for 'threshold' is not a number or 'auto'");
    }
  }
  if (const auto v = r.source("block")) {
    if (const auto* text = std::get_if<std::string>(&*v); text && *text == "auto") {
      s.block_mode = AuditConfig::BlockMode::politis_white;
    } else {
      s.block_mode = AuditConfig::BlockMode::fixed;
      s.block = *r.integer("block");
    }
  }
  if (const auto v = r.source("reported")) {
    if (const auto* text = std::get_if<std::string>(&*v)) {
      s.reported = *py_float(*text);
    } else if (const auto* d = std::get_if<double>(&*v)) {
      s.reported = *d;
    } else if (const auto* i = std::get_if<std::int64_t>(&*v)) {
      s.reported = static_cast<double>(*i);
      s.reported_is_int = true;
    } else {
      throw Unsupported("config value for 'reported' is not a number");
    }
  }
  return s;
}

// --- reading and reshaping the table ----------------------------------------------------------

std::vector<std::string> csv_list(const std::optional<std::string>& value) {
  std::vector<std::string> out;
  if (!value || value->empty()) return out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t comma = value->find(',', start);
    const std::string_view piece = py_strip(std::string_view(*value).substr(start, comma - start));
    if (!piece.empty()) out.emplace_back(piece);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

/// Path(text).suffix.lower()
std::string suffix_of(const std::string& text) {
  const std::string name = fs::path(text).filename().string();
  const std::size_t dot = name.rfind('.');
  if (dot == std::string::npos || dot == 0 || dot + 1 == name.size()) return "";
  return lowered(name.substr(dot));
}

Table load_table(const std::string& data, io::Pandas pandas) {
  const fs::path path(data);
  std::error_code ec;
  if (!fs::exists(path, ec)) throw CliError{"data file not found: " + py_path(data)};
  const std::string suffix = suffix_of(data);
  if (suffix == ".parquet" || suffix == ".feather") {
    throw Unsupported("reading " + std::string(suffix == ".parquet" ? "Parquet" : "Feather") +
                      " files needs the Python package (pyarrow)");
  }
  static constexpr std::array<std::string_view, 7> kCompressed = {".gz",  ".bz2", ".zip", ".xz",
                                                                  ".zst", ".tar", ".tgz"};
  if (std::find(kCompressed.begin(), kCompressed.end(), suffix) != kCompressed.end()) {
    throw Unsupported("compressed input files need the Python package");
  }
  if (fs::is_directory(path, ec)) throw Unsupported(py_path(data) + " is a directory");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Unsupported("cannot open " + py_path(data));
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  try {
    return io::read_csv(text, suffix == ".tsv" ? '\t' : ',', pandas);
  } catch (const io::ParseError& e) {
    throw CliError{e.what()};
  }
}

Column tiled(const Column& c, std::size_t times, std::string name) {
  Column out;
  out.name = std::move(name);
  out.kind = c.kind;
  for (std::size_t k = 0; k < times; ++k) {
    out.ints.insert(out.ints.end(), c.ints.begin(), c.ints.end());
    out.floats.insert(out.floats.end(), c.floats.begin(), c.floats.end());
    out.strings.insert(out.strings.end(), c.strings.begin(), c.strings.end());
    out.missing.insert(out.missing.end(), c.missing.begin(), c.missing.end());
  }
  return out;
}

void put(Table& t, Column c) {  // dict assignment: replace in place, else append
  for (auto& existing : t.columns) {
    if (existing.name == c.name) {
      existing = std::move(c);
      return;
    }
  }
  t.columns.push_back(std::move(c));
}

// predictive_audit.reshape.wide_to_long(df, time=time, keep=keep)
Table wide_to_long(const Table& t, const std::optional<std::string>& time,
                   const std::vector<std::string>& keep) {
  std::vector<std::string> fixed = keep;
  if (time) fixed.push_back(*time);
  std::vector<const Column*> cols;
  for (const auto& c : t.columns) {
    if (std::find(fixed.begin(), fixed.end(), c.name) == fixed.end()) cols.push_back(&c);
  }
  if (cols.empty()) throw CliError{"no value columns found to reshape"};
  const std::size_t n = t.n_rows;
  const std::size_t k = cols.size();

  Table out;
  out.n_rows = n * k;
  Column series;
  series.name = "series";
  Column step;
  step.name = "step";
  step.kind = ColumnKind::int64;
  for (const Column* c : cols) {
    series.strings.insert(series.strings.end(), n, c->name);
    for (std::size_t r = 0; r < n; ++r) step.ints.push_back(static_cast<std::int64_t>(r));
  }
  series.missing.assign(n * k, 0);
  put(out, std::move(series));
  put(out, std::move(step));
  for (const auto& name : fixed) {
    const Column* c = t.find(name);
    if (c == nullptr) throw CliError{py_repr_str(name)};  // str(KeyError(name))
    put(out, tiled(*c, k, name));
  }

  Column value;
  value.name = "value";
  const bool all_int =
      std::all_of(cols.begin(), cols.end(), [](const Column* c) { return c->kind == ColumnKind::int64; });
  const bool all_bool =
      std::all_of(cols.begin(), cols.end(), [](const Column* c) { return c->kind == ColumnKind::boolean; });
  if (n > 0 &&
      std::any_of(cols.begin(), cols.end(), [](const Column* c) { return c->kind == ColumnKind::string; })) {
    throw Unsupported("--wide needs every series column to be numeric");
  }
  if (all_int || all_bool) {
    value.kind = all_int ? ColumnKind::int64 : ColumnKind::boolean;
    for (const Column* c : cols) value.ints.insert(value.ints.end(), c->ints.begin(), c->ints.end());
  } else {
    value.kind = ColumnKind::float64;
    for (const Column* c : cols) {
      if (c->kind == ColumnKind::float64) {
        value.floats.insert(value.floats.end(), c->floats.begin(), c->floats.end());
      } else {
        for (const auto v : c->ints) value.floats.push_back(static_cast<double>(v));
        if (c->kind == ColumnKind::string) value.floats.resize(value.floats.size() + c->strings.size());
      }
    }
  }
  put(out, std::move(value));
  return out;
}

// --- building the engine's inputs --------------------------------------------------------------

/// Owns the arrays the AuditColumns spans point into.
struct Inputs {
  std::deque<std::vector<double>> reals;
  std::deque<std::vector<std::int64_t>> ints;
  std::deque<std::vector<std::uint8_t>> bytes;
  AuditColumns columns;
};

/// series.astype(float)
std::span<const double> numeric(Inputs& in, const Column& c) {
  auto& out = in.reals.emplace_back();
  switch (c.kind) {
    case ColumnKind::int64:
    case ColumnKind::boolean:
      out.assign(c.ints.begin(), c.ints.end());
      for (std::size_t i = 0; i < c.ints.size(); ++i) out[i] = static_cast<double>(c.ints[i]);
      break;
    case ColumnKind::float64:
      out = c.floats;
      break;
    case ColumnKind::string:
      if (!c.strings.empty()) {
        throw Unsupported("column " + py_repr_str(c.name) +
                          " holds text; the native engine reads only numeric "
                          "columns as numbers");
      }
      break;
  }
  return out;
}

/// predictive_audit.audit._key_column: numbers as they are, text as sorted factor codes.
KeyColumn key(Inputs& in, const Column& c) {
  switch (c.kind) {
    case ColumnKind::int64:
    case ColumnKind::boolean:
      return std::span<const std::int64_t>(in.ints.emplace_back(c.ints));
    case ColumnKind::float64:
      if (std::any_of(c.floats.begin(), c.floats.end(), [](double v) { return std::isnan(v); })) {
        throw Unsupported("column " + py_repr_str(c.name) + " has missing values");
      }
      return std::span<const double>(in.reals.emplace_back(c.floats));
    case ColumnKind::string:
      break;
  }
  if (std::any_of(c.missing.begin(), c.missing.end(), [](std::uint8_t m) { return m != 0; })) {
    throw Unsupported("column " + py_repr_str(c.name) + " has missing values");
  }
  std::vector<std::string> uniques(c.strings);
  std::sort(uniques.begin(), uniques.end());  // byte order of UTF-8 = code point order, as Python sorts
  uniques.erase(std::unique(uniques.begin(), uniques.end()), uniques.end());
  auto& codes = in.ints.emplace_back(c.strings.size());
  for (std::size_t i = 0; i < c.strings.size(); ++i) {
    codes[i] = std::lower_bound(uniques.begin(), uniques.end(), c.strings[i]) - uniques.begin();
  }
  return std::span<const std::int64_t>(codes);
}

/// predictive_audit.audit._split_kinds: astype(str).str.lower() == "train" / "test".
std::span<const std::uint8_t> split_kinds(Inputs& in, const Column& c) {
  auto& out = in.bytes.emplace_back(c.size(), kSplitOther);
  if (c.kind != ColumnKind::string) return out;  // str() of a number or bool is never train/test
  for (std::size_t i = 0; i < c.strings.size(); ++i) {
    if (c.missing[i] != 0) continue;
    const std::string label = lowered(c.strings[i]);  // only ASCII letters can lower to these
    out[i] = label == "test" ? kSplitTest : label == "train" ? kSplitTrain : kSplitOther;
  }
  return out;
}

std::string py_list(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t k = 0; k < items.size(); ++k) out += (k ? ", " : "") + py_repr_str(items[k]);
  return out + "]";
}

std::string py_tuple(std::span<const std::string_view> items) {
  std::string out = "(";
  for (std::size_t k = 0; k < items.size(); ++k) out += (k ? ", " : "") + py_repr_str(items[k]);
  return out + ")";
}

constexpr std::array<std::string_view, 7> kClassificationMetrics = {
    "f1", "pr_auc", "mcc", "balanced_accuracy", "accuracy", "roc_auc", "brier"};
constexpr std::array<std::string_view, 4> kRegressionMetrics = {"mae", "rmse", "r2", "smape"};

struct Outcome {
  int exit_code = 0;
  std::string out;
};

Outcome run_audit(const Settings& s, const Options& opts) {
  Table table = load_table(s.data, opts.pandas);
  std::vector<std::string> groups = csv_list(s.group);
  std::optional<std::string> time_col = s.time;
  std::optional<std::string> target_col = s.target;
  if (s.wide) {
    std::vector<std::string> keep = csv_list(s.wide_keep);
    if (s.split && std::find(keep.begin(), keep.end(), *s.split) == keep.end()) keep.push_back(*s.split);
    table = wide_to_long(table, time_col, keep);
    if (!target_col || target_col->empty()) target_col = "value";
    if (groups.empty()) groups = {"series"};
    if (!time_col || time_col->empty()) time_col = "step";
  }
  if (!target_col || target_col->empty()) throw CliError{"--target is required unless --wide is set"};
  const std::vector<std::string> features = csv_list(s.features);

  // predictive_audit.audit.audit: argument checks, in its order.
  if (s.task != "classification" && s.task != "regression") {
    throw CliError{"task must be 'classification' or 'regression'"};
  }
  if (s.horizon < 1) throw CliError{"horizon must be >= 1"};
  std::vector<std::string> missing;
  const auto require = [&](const std::optional<std::string>& c) {
    if (c && table.find(*c) == nullptr) missing.push_back(*c);
  };
  require(target_col);
  require(s.pred);
  require(time_col);
  require(s.split);
  for (const auto& g : groups) require(g);
  for (const auto& f : features) require(f);
  if (!missing.empty()) throw CliError{"columns not found: " + py_list(missing)};
  const bool clf = s.task == "classification";
  const std::string metric = (s.metric && !s.metric->empty()) ? *s.metric : (clf ? "f1" : "mae");
  const std::span<const std::string_view> allowed =
      clf ? std::span<const std::string_view>(kClassificationMetrics)
          : std::span<const std::string_view>(kRegressionMetrics);
  if (std::find(allowed.begin(), allowed.end(), metric) == allowed.end()) {
    throw CliError{"metric " + py_repr_str(metric) + " is not valid for " + s.task + "; choose from " +
                   py_tuple(allowed)};
  }

  // Arguments the pandas path reads differently from the engine (predictive_audit.audit._native_prepare).
  if (s.seed < 0) throw Unsupported("negative seeds");
  if (s.window && *s.window < 0) throw Unsupported("a negative --window");
  if (s.season_mode == AuditConfig::SeasonMode::fixed && s.season < 0)
    throw Unsupported("a negative --season");
  if (s.block_mode == AuditConfig::BlockMode::fixed && s.block < 1) throw Unsupported("--block below 1");

  Inputs in;
  const bool has_pred = s.pred && !s.pred->empty();
  in.columns.target = numeric(in, *table.find(*target_col));
  if (has_pred) in.columns.pred = numeric(in, *table.find(*s.pred));
  for (const auto& g : groups) in.columns.group_keys.push_back(key(in, *table.find(g)));
  if (time_col && !time_col->empty()) in.columns.time = key(in, *table.find(*time_col));
  if (s.split && !s.split->empty()) in.columns.split = split_kinds(in, *table.find(*s.split));
  for (const auto& f : features) in.columns.features.push_back({f, numeric(in, *table.find(f))});
  if (clf && in.columns.pred &&
      std::any_of(in.columns.pred->begin(), in.columns.pred->end(), [](double v) { return std::isinf(v); })) {
    throw Unsupported("infinite classification scores");  // scikit-learn rejects them with its own message
  }

  AuditConfig cfg;
  cfg.task = clf ? Task::classification : Task::regression;
  cfg.horizon = s.horizon;
  cfg.target_name = *target_col;
  if (has_pred) cfg.model_name = (s.model_name && !s.model_name->empty()) ? *s.model_name : *s.pred;
  cfg.split_name = s.split.value_or("");
  cfg.metric = *parse_metric(metric);
  cfg.season_mode = s.season_mode;
  cfg.season = s.season;
  cfg.window = s.window;
  cfg.include_ewma = s.ewma;
  cfg.ar_lags = s.ar_lags;
  cfg.reported = s.reported;
  cfg.threshold_auto = clf && s.threshold_auto;
  cfg.threshold = (clf && !s.threshold_auto) ? s.threshold : 0.5;
  cfg.n_boot = s.n_boot;
  cfg.block_mode = s.block_mode;
  cfg.block = s.block;
  cfg.seed = static_cast<std::uint64_t>(s.seed);
  cfg.ewm_com1_update = opts.ewm_com1_update;

  AuditReport report;
  try {
    report = audit(in.columns, cfg);
  } catch (const std::invalid_argument& e) {
    throw CliError{e.what()};
  }
  report.model = has_pred ? cfg.model_name : s.model_name;  // Python keeps model_name without --pred
  if (s.reported_is_int) {
    for (auto& c : report.checks) {
      if (c.name != "reported score") continue;
      for (auto& [k, v] : c.details) {
        if (k == "reported") v = static_cast<std::int64_t>(*s.reported);
      }
    }
  }

  std::string text;
  if (s.format == "text") {
    text = to_text(report);
  } else if (s.format == "markdown") {
    text = to_markdown(report);
  } else if (s.format == "json") {
    text = to_json(report);
  } else {
    throw CliError{"invalid --format " + py_repr_str(s.format)};
  }
  const auto verdict = static_cast<int>(report.verdict());
  const int exit_code =
      s.fail_on == "never"
          ? 0
          : (verdict >= static_cast<int>(s.fail_on == "fail" ? Status::fail : Status::warn) ? 1 : 0);
  if (s.out) {
    const fs::path out_path(*s.out);
    std::error_code ec;
    if (out_path.has_parent_path()) fs::create_directories(out_path.parent_path(), ec);
    std::ofstream f(out_path);  // text mode, like Path.write_text
    f << text << '\n';
    if (!f) throw CliError{"cannot write " + py_path(*s.out)};
    return {exit_code, ""};
  }
  return {exit_code, text + "\n"};
}

}  // namespace

Result run(const std::vector<std::string>& args, const Options& options) {
  Result res;
  try {
    const Parsed parsed = parse_args(args);
    if (parsed.help) {
      res.out = std::string(kUsage) + std::string(kHelpBody);
      return res;
    }
    if (parsed.version) {
      res.out = "predictive-audit " + std::string(version()) + "\n";
      return res;
    }
    if (!parsed.data) throw UsageError{"the following arguments are required: data"};
    std::map<std::string, Default, std::less<>> defaults;
    if (const auto it = parsed.values.find("config"); it != parsed.values.end())
      defaults = load_toml(it->second);
    if (!parsed.unrecognized.empty()) {
      std::string list;
      for (std::size_t k = 0; k < parsed.unrecognized.size(); ++k)
        list += (k ? " " : "") + parsed.unrecognized[k];
      throw UsageError{"unrecognized arguments: " + list};
    }
    const Settings settings = resolve(parsed, defaults);
    if (settings.threads) {
      if (*settings.threads < 1) throw CliError{"--threads must be >= 1"};
      parallel::set_num_threads(static_cast<unsigned>(std::min<std::int64_t>(*settings.threads, 1024)));
    }
    Outcome outcome = run_audit(settings, options);
    res.exit_code = outcome.exit_code;
    res.out = std::move(outcome.out);
  } catch (const UsageError& e) {
    res = {};
    res.exit_code = kExitError;
    res.err = std::string(kUsage) + "predictive-audit: error: " + e.message + "\n";
  } catch (const CliError& e) {
    res = {};
    res.exit_code = kExitError;
    res.err = "predictive-audit: error: " + e.message + "\n";
  } catch (const Unsupported& e) {
    res = {};
    if (options.allow_fallback) {
      res.fallback = true;
      res.fallback_reason = e.what();
    } else {
      res.exit_code = kExitError;
      res.err = std::string("predictive-audit: error: not supported by the native engine: ") + e.what() +
                " (the Python package handles it: pip install predictive-audit)\n";
    }
  }
  return res;
}

}  // namespace paudit::cli
