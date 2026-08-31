//  Program:      nes-py
//  File:         emulator_pool.hpp
//  Description:  A pool of NES emulators stepped in parallel by worker threads
//

#ifndef EMULATOR_POOL_HPP
#define EMULATOR_POOL_HPP

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "common.hpp"
#include "emulator.hpp"

namespace NES {

/// A pool of independent emulator instances (all running the same ROM) with a
/// persistent worker thread pool. Designed for planning workloads: batches of
/// states are loaded, stepped, and dumped in a single parallel operation.
///
/// State rows use a fixed stride of max_state_size() bytes. Emulator dumps may
/// be up to 8 bytes shorter than the stride (the state format is
/// self-describing, so a padded row loads correctly); trailing padding bytes
/// in a row are left untouched by dump_states.
class EmulatorPool {
 public:
    /// Initialize a pool.
    ///
    /// @param rom_path path to the ROM every emulator in the pool runs
    /// @param num_envs number of emulator instances
    /// @param num_workers number of worker threads; 0 selects
    ///        hardware_concurrency clamped to num_envs
    /// @param screen_in_state whether serialized states carry the screen
    ///        frame buffer (see Emulator); off by default for small states
    ///
    EmulatorPool(const std::string& rom_path, int num_envs, int num_workers,
                 bool screen_in_state = false);

    /// Shut down the workers and delete the emulators.
    ~EmulatorPool();

    /// Return the number of emulator instances in the pool.
    inline int num_envs() const { return static_cast<int>(emulators.size()); }

    /// Return the fixed state row stride in bytes.
    inline size_t max_state_size() const { return stride; }

    /// Return a pointer to emulator i (for per-instance operations).
    inline Emulator* get(int i) { return emulators[i]; }

    /// Reset every emulator in the pool.
    void reset_all();

    /// Reset a single emulator.
    inline void reset_one(int i) { emulators[i]->reset(); }

    /// Perform a fused batch operation on every emulator in parallel. For
    /// each emulator i the following phases run in order, each skipped when
    /// its pointer is null:
    ///
    /// 1. load_state from states_in + i * stride
    /// 2. write actions[i] to controller 0 and step dt[i] * frameskip frames
    ///    (skipped entirely when actions or dt is null)
    /// 3. dump_state to states_out + i * stride
    /// 4. copy the 240x256 screen as RGB bytes to screens_out + i * 240*256*3
    /// 5. copy the 2KB work RAM to ram_out + i * 0x800
    ///
    void step_batch(const NES_Byte* actions, const int32_t* dt, int frameskip,
                    const char* states_in, char* states_out,
                    uint8_t* screens_out, NES_Byte* ram_out);

    /// Load a state into every emulator from a (num_envs, stride) buffer.
    inline void load_states(const char* states) {
        step_batch(nullptr, nullptr, 1, states, nullptr, nullptr, nullptr);
    }

    /// Dump every emulator's state into a (num_envs, stride) buffer.
    inline void dump_states(char* states) {
        step_batch(nullptr, nullptr, 1, nullptr, states, nullptr, nullptr);
    }

    /// Copy every screen as RGB uint8 into a (num_envs, 240, 256, 3) buffer.
    inline void read_screens(uint8_t* out) {
        step_batch(nullptr, nullptr, 1, nullptr, nullptr, out, nullptr);
    }

    /// Copy every emulator's work RAM into a (num_envs, 0x800) buffer.
    inline void read_ram(NES_Byte* out) {
        step_batch(nullptr, nullptr, 1, nullptr, nullptr, nullptr, out);
    }

 private:
    /// the emulator instances, one per batch element
    std::vector<Emulator*> emulators;
    /// the fixed state row stride (= emulators[0]->max_state_size())
    size_t stride = 0;

    /// the number of worker threads (fixed before any thread starts)
    int num_workers = 0;
    /// the persistent worker threads
    std::vector<std::thread> workers;
    /// guards generation / pending / task / shutting_down
    std::mutex mtx;
    /// signaled when a new task generation is published
    std::condition_variable cv_start;
    /// signaled when the last worker finishes a generation
    std::condition_variable cv_done;
    /// incremented once per dispatched parallel operation
    uint64_t generation = 0;
    /// number of workers that have not finished the current generation
    int pending = 0;
    /// set by the destructor to stop the worker loops
    bool shutting_down = false;
    /// the per-emulator-index task for the current generation
    std::function<void(int)> task;

    /// Run fn(i) for every emulator index i across the worker threads and
    /// block until all of them complete.
    void run_parallel(const std::function<void(int)>& fn);

    /// The worker thread main loop; worker w owns the contiguous slice of
    /// emulator indices [w * n / W, (w + 1) * n / W).
    void worker_loop(int worker_id);
};

}  // namespace NES

#endif  // EMULATOR_POOL_HPP
