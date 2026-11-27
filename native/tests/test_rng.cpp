// Golden values captured from NumPy 2.5 (np.random.default_rng / SeedSequence / integers).
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include "paudit/rng.hpp"

using paudit::rng::Pcg64;
using paudit::rng::U128;

TEST_CASE("128-bit multiply matches the portable definition", "[rng]") {
  const U128 p = paudit::rng::mul_64x64(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL);
  CHECK(p.hi == 0xFFFFFFFFFFFFFFFEULL);
  CHECK(p.lo == 0x0000000000000001ULL);
  const U128 s = paudit::rng::add(U128{0, 0xFFFFFFFFFFFFFFFFULL}, U128{0, 1});
  CHECK(s == U128{1, 0});
}

TEST_CASE("SeedSequence pool matches NumPy for a two-word seed", "[rng]") {
  const paudit::rng::SeedSequence ss((std::uint64_t{1} << 40) + 17);
  const std::array<std::uint32_t, 4> expected{3329045468U, 3165939009U, 3179801948U, 3310054689U};
  CHECK(ss.pool() == expected);
}

TEST_CASE("PCG64 seeding matches np.random.default_rng", "[rng]") {
  const Pcg64 g0(0);
  CHECK(g0.state() == U128{0x1aa1b5345996452dULL, 0x09585eb7a69561e3ULL});
  CHECK(g0.inc() == U128{0x418ddadb3af71a82ULL, 0x588133bc447873a9ULL});
  const Pcg64 g(12345);
  CHECK(g.state() == U128{0x1905e0335aae9634ULL, 0x9199b0d09775add5ULL});
  CHECK(g.inc() == U128{0xc9c7353e6e2b1f28ULL, 0x7d761f2d4027fae7ULL});
}

TEST_CASE("raw 64-bit outputs match bit_generator.random_raw", "[rng]") {
  Pcg64 g(7);
  const std::array<std::uint64_t, 5> expected{11530976094092348043ULL, 16550673365885938325ULL,
                                              14308875409591826786ULL, 4154339397315733314ULL,
                                              5537090637313560901ULL};
  for (const auto v : expected) CHECK(g.next64() == v);
}

TEST_CASE("bounded integers match Generator.integers", "[rng]") {
  SECTION("32-bit Lemire path") {
    Pcg64 g(12345);
    std::vector<std::int64_t> out(12);
    paudit::rng::integers(g, 317, out);
    CHECK(out == std::vector<std::int64_t>{221, 72, 250, 100, 64, 252, 203, 214, 313, 123, 266, 105});
  }
  SECTION("split calls consume the same stream as one call") {
    Pcg64 a(12345);
    Pcg64 b(12345);
    std::vector<std::int64_t> whole(12);
    std::vector<std::int64_t> first(5);
    std::vector<std::int64_t> second(7);
    paudit::rng::integers(a, 317, whole);
    paudit::rng::integers(b, 317, first);
    paudit::rng::integers(b, 317, second);
    first.insert(first.end(), second.begin(), second.end());
    CHECK(first == whole);
  }
  SECTION("full 32-bit range") {
    Pcg64 g(3);
    std::vector<std::int64_t> out(4);
    paudit::rng::integers(g, std::uint64_t{1} << 32, out);
    CHECK(out == std::vector<std::int64_t>{3485385464, 367860371, 770692085, 1017093381});
  }
  SECTION("64-bit Lemire path") {
    Pcg64 g(3);
    std::vector<std::int64_t> out(4);
    paudit::rng::integers(g, 5'000'000'000ULL, out);
    CHECK(out == std::vector<std::int64_t>{428245835, 1184052532, 4006372326, 2910810180});
  }
  SECTION("high == 1 draws nothing") {
    Pcg64 g(1);
    const U128 before = g.state();
    std::vector<std::int64_t> out(3, 9);
    paudit::rng::integers(g, 1, out);
    CHECK(out == std::vector<std::int64_t>{0, 0, 0});
    CHECK(g.state() == before);
  }
}
