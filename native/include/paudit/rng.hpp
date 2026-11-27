#pragma once

// Bit-exact reimplementation of NumPy's default generator stack:
//   np.random.default_rng(seed)  ==  Generator(PCG64(SeedSequence(seed)))
// and of `Generator.integers(0, high, size)` (Lemire's bounded rejection, the default
// unmasked path). Same seed, same draws, so native bootstrap intervals reproduce the
// Python reference exactly.

#include <array>
#include <cstdint>
#include <span>

namespace paudit::rng {

/// 128-bit unsigned integer with the two operations PCG64 needs (portable; no __int128 required).
struct U128 {
  std::uint64_t hi = 0;
  std::uint64_t lo = 0;

  friend constexpr bool operator==(const U128&, const U128&) = default;
};

/// Full 64x64 -> 128-bit product.
[[nodiscard]] U128 mul_64x64(std::uint64_t a, std::uint64_t b) noexcept;
/// Low 128 bits of a 128x128-bit product.
[[nodiscard]] U128 mul(U128 a, U128 b) noexcept;
/// a + b modulo 2^128.
[[nodiscard]] U128 add(U128 a, U128 b) noexcept;

/// numpy.random.SeedSequence for a non-negative integer seed with no spawn key.
class SeedSequence {
 public:
  explicit SeedSequence(std::uint64_t entropy);

  /// `generate_state(n_words, np.uint32)`.
  void generate_state(std::span<std::uint32_t> out) const;
  /// `generate_state(n_words, np.uint64)` (little-endian pairs of uint32 words).
  void generate_state(std::span<std::uint64_t> out) const;

  [[nodiscard]] const std::array<std::uint32_t, 4>& pool() const noexcept { return pool_; }

 private:
  std::array<std::uint32_t, 4> pool_{};
};

/// numpy.random.PCG64 (128-bit LCG, XSL-RR output) including its buffered 32-bit draws.
class Pcg64 {
 public:
  /// Seeded exactly like `np.random.PCG64(seed)` / `np.random.default_rng(seed)`.
  explicit Pcg64(std::uint64_t seed);
  /// Raw state, as reported by `bit_generator.state`.
  Pcg64(U128 state, U128 inc) noexcept : state_(state), inc_(inc) {}

  [[nodiscard]] std::uint64_t next64() noexcept;
  [[nodiscard]] std::uint32_t next32() noexcept;

  [[nodiscard]] U128 state() const noexcept { return state_; }
  [[nodiscard]] U128 inc() const noexcept { return inc_; }

 private:
  void step() noexcept;

  U128 state_{};
  U128 inc_{};
  bool has_uint32_ = false;
  std::uint32_t uinteger_ = 0;
};

/// `Generator.integers(0, high, size=out.size())` with dtype int64 (high >= 1).
void integers(Pcg64& gen, std::uint64_t high, std::span<std::int64_t> out);

}  // namespace paudit::rng
