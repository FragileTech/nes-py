//  Program:      nes-py
//  File:         emulator_pool.cpp
//  Description:  A pool of NES emulators stepped in parallel by worker threads
//

#include "emulator_pool.hpp"
#include <cstring>
#include "ppu.hpp"

namespace NES {

EmulatorPool::EmulatorPool(const std::string& rom_path, int num_envs,
        int num_workers, bool screen_in_state) {
    if (num_envs < 1) num_envs = 1;
    emulators.reserve(num_envs);
    for (int i = 0; i < num_envs; i++)
        emulators.push_back(new Emulator(rom_path, screen_in_state));
    stride = emulators[0]->max_state_size();
    if (num_workers <= 0)
        num_workers = static_cast<int>(std::thread::hardware_concurrency());
    if (num_workers < 1) num_workers = 1;
    if (num_workers > num_envs) num_workers = num_envs;
    this->num_workers = num_workers;
    workers.reserve(num_workers);
    for (int w = 0; w < num_workers; w++)
        workers.emplace_back(&EmulatorPool::worker_loop, this, w);
}

EmulatorPool::~EmulatorPool() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
    }
    cv_start.notify_all();
    for (auto& worker : workers)
        worker.join();
    for (auto* emulator : emulators)
        delete emulator;
}

void EmulatorPool::worker_loop(int worker_id) {
    const int n = num_envs();
    const int begin = worker_id * n / num_workers;
    const int end = (worker_id + 1) * n / num_workers;
    uint64_t last_generation = 0;
    while (true) {
        std::function<void(int)> local_task;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_start.wait(lock, [&] {
                return shutting_down || generation != last_generation;
            });
            if (shutting_down) return;
            last_generation = generation;
            local_task = task;
        }
        for (int i = begin; i < end; i++) {
            // never let an exception escape the worker thread
            try { local_task(i); } catch (...) { }
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (--pending == 0) cv_done.notify_all();
        }
    }
}

void EmulatorPool::run_parallel(const std::function<void(int)>& fn) {
    std::unique_lock<std::mutex> lock(mtx);
    task = fn;
    pending = num_workers;
    generation++;
    cv_start.notify_all();
    cv_done.wait(lock, [&] { return pending == 0; });
}

void EmulatorPool::reset_all() {
    run_parallel([this](int i) { emulators[i]->reset(); });
}

void EmulatorPool::step_batch(const NES_Byte* actions, const int32_t* dt,
        int frameskip, const char* states_in, char* states_out,
        uint8_t* screens_out, NES_Byte* ram_out) {
    if (frameskip < 1) frameskip = 1;
    const size_t row = stride;
    run_parallel([=](int i) {
        Emulator* emulator = emulators[i];
        if (states_in != nullptr)
            emulator->load_state(states_in + i * row);
        if (actions != nullptr && dt != nullptr) {
            *emulator->get_controller(0) = actions[i];
            const int frames = dt[i] * frameskip;
            for (int f = 0; f < frames; f++)
                emulator->step();
        }
        if (states_out != nullptr)
            emulator->dump_state(states_out + i * row);
        if (screens_out != nullptr) {
            // convert the native 32-bit xRGB frame buffer to packed RGB bytes
            const NES_Pixel* pixels = emulator->get_screen_buffer();
            uint8_t* out = screens_out
                + static_cast<size_t>(i) * Emulator::HEIGHT * Emulator::WIDTH * 3;
            const int num_pixels = Emulator::HEIGHT * Emulator::WIDTH;
            for (int p = 0; p < num_pixels; p++) {
                const NES_Pixel pixel = pixels[p];
                out[3 * p + 0] = static_cast<uint8_t>(pixel >> 16);
                out[3 * p + 1] = static_cast<uint8_t>(pixel >> 8);
                out[3 * p + 2] = static_cast<uint8_t>(pixel);
            }
        }
        if (ram_out != nullptr)
            std::memcpy(ram_out + i * 0x800, emulator->get_memory_buffer(), 0x800);
    });
}

}  // namespace NES
