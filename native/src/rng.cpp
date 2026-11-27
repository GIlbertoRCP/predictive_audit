#include "paudit/rng.hpp"

#include <bit>
#include <cstddef>
#include <limits>
#include <stdexcept>

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

namespace paudit::rng {

// --- 128-bit arithmetic -------------------------------------------------------

U128 mul_64x64(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
  __extension__ using u128_native = unsigned __int128;
  const u128_native p = static_cast<u128_native>(a) * static_cast<u128_native>(b);
  return U128{static_cast<std::uint64_t>(p >> 64), static_cast<std::uint64_t>(p)};
#elif defined(_MSC_VER) && defined(_M_X64)
  std::uint64_t hi = 0;
  const std::uint64_t lo = _umul128(a, b, &hi);
  return U128{hi, lo};
#elif defined(_MSC_VER) && defined(_M_ARM64)
  return U128{__umulh(a, b), a * b};
#else
  const std::uint64_t a_lo = a & 0xFFFFFFFFULL;
  const std::uint64_t a_hi = a >> 32;
  const std::uint64_t b_lo = b & 0xFFFFFFFFULL;
  const std::uint64_t b_hi = b >> 32;
  const std::uint64_t ll = a_lo * b_lo;
  const std::uint64_t lh = a_lo * b_hi;
  const std::uint64_t hl = a_hi * b_lo;
  const std::uint64_t hh = a_hi * b_hi;
  const std::uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFULL) + (hl & 0xFFFFFFFFULL);
  return U128{hh + (lh >> 32) + (hl >> 32) + (mid >> 32), (mid << 32) | (ll & 0xFFFFFFFFULL)};
#endif
}

U128 mul(U128 a, U128 b) noexcept {
  U128 r = mul_64x64(a.lo, b.lo);
  r.hi += a.hi * b.lo + a.lo * b.hi;
  return r;
}

U128 add(U128 a, U128 b) noexcept {
  U128 r{a.hi + b.hi, a.lo + b.lo};
  if (r.lo < a.lo) ++r.hi;
  return r;
}

// --- SeedSequence -------------------------------------------------------------

namespace {

constexpr std::uint32_t kInitA = 0x43b0d7e5U;
constexpr std::uint32_t kMultA = 0x931e8875U;
constexpr std::uint32_t kInitB = 0x8b51f9ddU;
constexpr std::uint32_t kMultB = 0x58f38dedU;
constexpr std::uint32_t kMixMultL = 0xca01f9ddU;
constexpr std::uint32_t kMixMultR = 0x4973f715U;
constexpr unsigned kXShift = 16;
constexpr std::size_t kPoolSize = 4;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) noexcept {
  value ^= hash_const;
  hash_const *= kMultA;
  value *= hash_const;
  value ^= value >> kXShift;
  return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) noexcept {
  std::uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

}  // namespace

SeedSequence::SeedSequence(std::uint64_t entropy) {
  // _int_to_uint32_array: little-endian 32-bit words, [0] for zero.
  std::array<std::uint32_t, 2> words{};
  std::size_t n_words = 0;
  if (entropy == 0) {
    words[n_words++] = 0;
  }
  while (entropy > 0) {
    words[n_words++] = static_cast<std::uint32_t>(entropy & 0xFFFFFFFFULL);
    entropy >>= 32;
  }

  std::uint32_t hash_const = kInitA;
  for (std::size_t i = 0; i < kPoolSize; ++i) {
    pool_[i] = hashmix(i < n_words ? words[i] : 0U, hash_const);
  }
  for (std::size_t i_src = 0; i_src < kPoolSize; ++i_src) {
    for (std::size_t i_dst = 0; i_dst < kPoolSize; ++i_dst) {
      if (i_src != i_dst) pool_[i_dst] = mix(pool_[i_dst], hashmix(pool_[i_src], hash_const));
    }
  }
  // At most two entropy words, never more than the pool size: no remaining entropy to fold in.
}

void SeedSequence::generate_state(std::span<std::uint32_t> out) const {
  std::uint32_t hash_const = kInitB;
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint32_t data = pool_[i % kPoolSize];
    data ^= hash_const;
    hash_const *= kMultB;
    data *= hash_const;
    data ^= data >> kXShift;
    out[i] = data;
  }
}

void SeedSequence::generate_state(std::span<std::uint64_t> out) const {
  std::array<std::uint32_t, 8> words{};
  if (out.size() * 2 > words.size()) throw std::invalid_argument("generate_state: at most 4 uint64 words");
  generate_state(std::span<std::uint32_t>(words.data(), out.size() * 2));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<std::uint64_t>(words[2 * i]) | (static_cast<std::uint64_t>(words[2 * i + 1]) << 32);
  }
}

// --- PCG64 ----------------------------------------------------------------------

namespace {
constexpr U128 kPcgMultiplier{0x2360ED051FC65DA4ULL, 0x4385DF649FCCF645ULL};
}  // namespace

Pcg64::Pcg64(std::uint64_t seed) {
  std::array<std::uint64_t, 4> val{};
  SeedSequence(seed).generate_state(std::span<std::uint64_t>(val));
  const U128 initstate{val[0], val[1]};
  const U128 initseq{val[2], val[3]};
  // pcg_setseq_128_srandom_r
  state_ = U128{};
  inc_ = U128{(initseq.hi << 1) | (initseq.lo >> 63), (initseq.lo << 1) | 1ULL};
  step();
  state_ = add(state_, initstate);
  step();
}

void Pcg64::step() noexcept { state_ = add(mul(state_, kPcgMultiplier), inc_); }

std::uint64_t Pcg64::next64() noexcept {
  step();
  const auto rot = static_cast<int>(state_.hi >> 58U);
  return std::rotr(state_.hi ^ state_.lo, rot);
}

std::uint32_t Pcg64::next32() noexcept {
  if (has_uint32_) {
    has_uint32_ = false;
    return uinteger_;
  }
  const std::uint64_t next = next64();
  has_uint32_ = true;
  uinteger_ = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xFFFFFFFFULL);
}

// --- bounded integers -------------------------------------------------------------

namespace {

std::uint32_t bounded_lemire_uint32(Pcg64& gen, std::uint32_t rng) noexcept {
  const std::uint32_t rng_excl = rng + 1;
  std::uint64_t m = static_cast<std::uint64_t>(gen.next32()) * rng_excl;
  auto leftover = static_cast<std::uint32_t>(m & 0xFFFFFFFFULL);
  if (leftover < rng_excl) {
    const std::uint32_t threshold = (std::numeric_limits<std::uint32_t>::max() - rng) % rng_excl;
    while (leftover < threshold) {
      m = static_cast<std::uint64_t>(gen.next32()) * rng_excl;
      leftover = static_cast<std::uint32_t>(m & 0xFFFFFFFFULL);
    }
  }
  return static_cast<std::uint32_t>(m >> 32);
}

std::uint64_t bounded_lemire_uint64(Pcg64& gen, std::uint64_t rng) noexcept {
  const std::uint64_t rng_excl = rng + 1;
  U128 m = mul_64x64(gen.next64(), rng_excl);
  if (m.lo < rng_excl) {
    const std::uint64_t threshold = (std::numeric_limits<std::uint64_t>::max() - rng) % rng_excl;
    while (m.lo < threshold) m = mul_64x64(gen.next64(), rng_excl);
  }
  return m.hi;
}

}  // namespace

void integers(Pcg64& gen, std::uint64_t high, std::span<std::int64_t> out) {
  if (high == 0) throw std::invalid_argument("integers: high must be >= 1");
  if (high > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("integers: high exceeds int64 range");
  }
  const std::uint64_t rng = high - 1;
  if (rng == 0) {
    for (auto& v : out) v = 0;
  } else if (rng < 0xFFFFFFFFULL) {
    const auto rng32 = static_cast<std::uint32_t>(rng);
    for (auto& v : out) v = static_cast<std::int64_t>(bounded_lemire_uint32(gen, rng32));
  } else if (rng == 0xFFFFFFFFULL) {
    for (auto& v : out) v = static_cast<std::int64_t>(gen.next32());
  } else {
    for (auto& v : out) v = static_cast<std::int64_t>(bounded_lemire_uint64(gen, rng));
  }
}

}  // namespace paudit::rng
