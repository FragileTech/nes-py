//  Program:      nes-py
//  File:         lib_nes_env.cpp
//  Description:  file describes the outward facing ctypes API for Python
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//

#include <cstdint>
#include <string>
#include "common.hpp"
#include "emulator.hpp"
#include "emulator_pool.hpp"

// Windows-base systems
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
    // setup the module initializer. required to link visual studio C++ ctypes
    void PyInit_lib_nes_env() { }
    // setup the function modifier to export in the DLL
    #define EXP __declspec(dllexport)
// Unix-like systems
#else
    // setup the modifier as a dummy
    #define EXP
#endif

// definitions of functions for the Python interface to access
extern "C" {
    /// Return the width of the NES.
    EXP int Width() {
        return NES::Emulator::WIDTH;
    }

    /// Return the height of the NES.
    EXP int Height() {
        return NES::Emulator::HEIGHT;
    }

    /// Initialize a new emulator and return a pointer to it. States carry
    /// the screen frame buffer only when screen_in_state is nonzero (legacy
    /// format); by default states are ~55x smaller and the screen is redrawn
    /// after stepping one frame.
    EXP NES::Emulator* Initialize(wchar_t* path, int screen_in_state) {
        // convert the c string to a c++ std string data structure
        std::wstring ws_rom_path(path);
        std::string rom_path(ws_rom_path.begin(), ws_rom_path.end());
        // create a new emulator with the given ROM path
        return new NES::Emulator(rom_path, screen_in_state != 0);
    }

    /// Return a pointer to a controller on the machine
    EXP NES::NES_Byte* Controller(NES::Emulator* emu, int port) {
        return emu->get_controller(port);
    }

    /// Return the pointer to the screen buffer
    EXP NES::NES_Pixel* Screen(NES::Emulator* emu) {
        return emu->get_screen_buffer();
    }

    /// Return the pointer to the memory buffer
    EXP NES::NES_Byte* Memory(NES::Emulator* emu) {
        return emu->get_memory_buffer();
    }

    /// Reset the emulator
    EXP void Reset(NES::Emulator* emu) {
        emu->reset();
    }

    /// Perform a discrete step in the emulator (i.e., 1 frame)
    EXP void Step(NES::Emulator* emu) {
        emu->step();
    }

    /// Create a deep copy (i.e., a clone) of the given emulator
    EXP void Backup(NES::Emulator* emu) {
        emu->backup();
    }

    /// Create a deep copy (i.e., a clone) of the given emulator
    EXP void Restore(NES::Emulator* emu) {
        emu->restore();
    }

    /// Close the emulator, i.e., purge it from memory
    EXP void Close(NES::Emulator* emu) {
        delete emu;
    }

    EXP size_t StateSize(NES::Emulator* emu) noexcept {
        return emu->state_size();
    }

    EXP void DumpState(NES::Emulator* emu, void *buffer) {
        emu->dump_state(reinterpret_cast<char *>(buffer));
    }

     EXP void LoadState(NES::Emulator* emu, const void *buffer) {
        emu->load_state(reinterpret_cast<const char *>(buffer));
    }

    /// Return the maximum state size (constant for a given ROM); safe to use
    /// as a fixed buffer size across steps, unlike StateSize.
    EXP size_t MaxStateSize(NES::Emulator* emu) noexcept {
        return emu->max_state_size();
    }

    /// Initialize a pool of emulators with a persistent worker thread pool.
    /// screen_in_state selects the legacy state format (see Initialize).
    EXP NES::EmulatorPool* InitializePool(wchar_t* path, int num_envs,
            int num_workers, int screen_in_state) {
        std::wstring ws_rom_path(path);
        std::string rom_path(ws_rom_path.begin(), ws_rom_path.end());
        return new NES::EmulatorPool(
            rom_path, num_envs, num_workers, screen_in_state != 0);
    }

    /// Close the pool, i.e., purge it from memory
    EXP void ClosePool(NES::EmulatorPool* pool) {
        delete pool;
    }

    /// Return the number of emulators in the pool
    EXP int PoolNumEnvs(NES::EmulatorPool* pool) {
        return pool->num_envs();
    }

    /// Return the fixed state row stride for batch state buffers
    EXP size_t PoolMaxStateSize(NES::EmulatorPool* pool) {
        return pool->max_state_size();
    }

    /// Reset every emulator in the pool
    EXP void PoolResetAll(NES::EmulatorPool* pool) {
        pool->reset_all();
    }

    /// Reset a single emulator in the pool
    EXP void PoolReset(NES::EmulatorPool* pool, int index) {
        pool->reset_one(index);
    }

    /// Fused parallel batch operation; any pointer may be NULL to skip its
    /// phase. Per element i: load states_in row, apply actions[i] for
    /// dt[i] * frameskip frames, dump to states_out row, write RGB screen
    /// and 2KB RAM.
    EXP void PoolStepBatch(NES::EmulatorPool* pool, const void* actions,
            const void* dt, int frameskip, const void* states_in,
            void* states_out, void* screens_out, void* ram_out) {
        pool->step_batch(
            reinterpret_cast<const NES::NES_Byte*>(actions),
            reinterpret_cast<const int32_t*>(dt),
            frameskip,
            reinterpret_cast<const char*>(states_in),
            reinterpret_cast<char*>(states_out),
            reinterpret_cast<uint8_t*>(screens_out),
            reinterpret_cast<NES::NES_Byte*>(ram_out));
    }

    /// Load a state into every emulator from a (num_envs, stride) buffer
    EXP void PoolLoadStates(NES::EmulatorPool* pool, const void* states) {
        pool->load_states(reinterpret_cast<const char*>(states));
    }

    /// Dump every emulator's state into a (num_envs, stride) buffer
    EXP void PoolDumpStates(NES::EmulatorPool* pool, void* states) {
        pool->dump_states(reinterpret_cast<char*>(states));
    }

    /// Copy every screen as RGB uint8 into a (num_envs, 240, 256, 3) buffer
    EXP void PoolScreens(NES::EmulatorPool* pool, void* screens_rgb) {
        pool->read_screens(reinterpret_cast<uint8_t*>(screens_rgb));
    }

    /// Copy every emulator's work RAM into a (num_envs, 0x800) buffer
    EXP void PoolRAM(NES::EmulatorPool* pool, void* ram) {
        pool->read_ram(reinterpret_cast<NES::NES_Byte*>(ram));
    }
}

// un-define the macro
#undef EXP
