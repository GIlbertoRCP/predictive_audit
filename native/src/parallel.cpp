#include "paudit/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#define PAUDIT_HAS_FORK 1
#endif

namespace paudit::parallel {
namespace {

thread_local bool tl_inside_parallel = false;

unsigned default_threads() {
  if (const char* env = std::getenv("PREDICTIVE_AUDIT_THREADS"); env != nullptr && *env != '\0') {
    try {
      const long v = std::stol(env);
      if (v > 0) return static_cast<unsigned>(std::min<long>(v, 1024));
    } catch (const std::exception&) {
      // Ignore malformed values and fall back to the hardware default.
    }
  }
  return std::max(1U, std::thread::hardware_concurrency());
}

class Pool {
 public:
  explicit Pool(unsigned n_threads) : n_threads_(std::max(1U, n_threads)) {
    workers_.reserve(n_threads_ - 1);
    for (unsigned i = 0; i + 1 < n_threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  Pool(Pool&&) = delete;
  Pool& operator=(Pool&&) = delete;

  ~Pool() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
  }

  [[nodiscard]] unsigned size() const noexcept { return n_threads_; }

  void run(std::size_t n, std::size_t grain, const std::function<void(std::size_t, std::size_t)>& fn) {
    const std::lock_guard<std::mutex> submit(submit_mutex_);  // one job at a time
    {
      // Workers that woke late for the previous job must leave run_chunks before it is reset.
      std::unique_lock<std::mutex> lock(mutex_);
      done_.wait(lock, [this] { return active_ == 0; });
      fn_ = &fn;
      n_ = n;
      grain_ = grain;
      n_chunks_ = (n + grain - 1) / grain;
      next_chunk_.store(0, std::memory_order_relaxed);
      error_ = nullptr;
      ++generation_;
    }
    wake_.notify_all();

    tl_inside_parallel = true;
    run_chunks();
    tl_inside_parallel = false;

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return active_ == 0; });
    fn_ = nullptr;
    if (error_) std::rethrow_exception(error_);
  }

 private:
  void run_chunks() {
    for (;;) {
      const std::size_t c = next_chunk_.fetch_add(1, std::memory_order_relaxed);
      if (c >= n_chunks_) return;
      const std::size_t begin = c * grain_;
      const std::size_t end = std::min(n_, begin + grain_);
      try {
        (*fn_)(begin, end);
      } catch (...) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = std::current_exception();
      }
    }
  }

  void worker_loop() {
    std::uint64_t seen = 0;
    tl_inside_parallel = true;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
        ++active_;
      }
      run_chunks();
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        --active_;
      }
      done_.notify_all();
    }
  }

  unsigned n_threads_;
  std::vector<std::thread> workers_;
  std::mutex submit_mutex_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  bool stopping_ = false;
  std::uint64_t generation_ = 0;
  unsigned active_ = 0;

  const std::function<void(std::size_t, std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0;
  std::size_t grain_ = 1;
  std::size_t n_chunks_ = 0;
  std::atomic<std::size_t> next_chunk_{0};
  std::exception_ptr error_;
};

std::mutex g_pool_mutex;
// Callers of parallel_for hold their own reference, so replacing the pool (set_num_threads) never
// destroys one that is still running a job: the old pool is torn down when its last job ends.
std::shared_ptr<Pool> g_pool;      // guarded by g_pool_mutex
unsigned g_requested_threads = 0;  // 0 = default

#if defined(PAUDIT_HAS_FORK)
// fork() copies only the calling thread. In the child the workers are gone and the pool's mutexes
// may be held by threads that no longer exist, so the child must never touch the inherited pool.
void before_fork() { g_pool_mutex.lock(); }
void after_fork_in_parent() { g_pool_mutex.unlock(); }
void after_fork_in_child() {
  if (g_pool) {
    // Deliberately leaked: destroying it would join threads that do not exist in this process.
    static_cast<void>(new std::shared_ptr<Pool>(std::move(g_pool)));
  }
  tl_inside_parallel = false;
  g_pool_mutex.unlock();
}
#endif

void register_fork_handlers() {
#if defined(PAUDIT_HAS_FORK)
  static std::once_flag once;
  std::call_once(once, [] { pthread_atfork(before_fork, after_fork_in_parent, after_fork_in_child); });
#endif
}

std::shared_ptr<Pool> pool() {
  register_fork_handlers();
  const std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (!g_pool) {
    g_pool = std::make_shared<Pool>(g_requested_threads != 0 ? g_requested_threads : default_threads());
  }
  return g_pool;
}

}  // namespace

unsigned num_threads() { return pool()->size(); }

void set_num_threads(unsigned n) {
  std::shared_ptr<Pool> old;
  {
    const std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_requested_threads = n;
    old = std::move(g_pool);
  }
  // `old` is released outside the lock; if a job is still running on it, that job's caller
  // holds the last reference and the pool shuts down when the job finishes.
}

void parallel_for(std::size_t n, std::size_t grain, const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0) return;
  grain = std::max<std::size_t>(1, grain);
  if (n <= grain || tl_inside_parallel) {
    for (std::size_t begin = 0; begin < n; begin += grain) fn(begin, std::min(n, begin + grain));
    return;
  }
  const std::shared_ptr<Pool> p = pool();
  if (p->size() == 1) {
    for (std::size_t begin = 0; begin < n; begin += grain) fn(begin, std::min(n, begin + grain));
    return;
  }
  p->run(n, grain, fn);
}

}  // namespace paudit::parallel
