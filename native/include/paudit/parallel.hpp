#pragma once

// A small persistent thread pool. Work is split into index ranges and every result is written
// to its own slot, so output never depends on the number of threads or on scheduling order.

#include <cstddef>
#include <functional>

namespace paudit::parallel {

/// Threads used by parallel_for (including the calling thread). Defaults to the
/// PREDICTIVE_AUDIT_THREADS environment variable, else std::thread::hardware_concurrency().
[[nodiscard]] unsigned num_threads();

/// Set the thread count; 0 restores the default. Safe at any time: jobs already running finish on
/// the previous pool, which shuts down after its last job. The pool is also fork-safe: a child
/// process created with fork() starts a fresh pool instead of reusing the parent's.
void set_num_threads(unsigned n);

/// Call `fn(begin, end)` over disjoint ranges covering [0, n), each at most `grain` long.
/// Runs inline when n fits in one range, when only one thread is configured, or when called
/// from inside another parallel_for (no nested oversubscription). Rethrows the first exception.
void parallel_for(std::size_t n, std::size_t grain, const std::function<void(std::size_t, std::size_t)>& fn);

}  // namespace paudit::parallel
