"""A vectorized NES environment backed by a native C++ emulator pool.

The C++ layer owns ``num_envs`` emulator instances and a persistent worker
thread pool; batch state loading, stepping, state dumping, and screen/RAM
readout all happen in a single library call with the GIL released. The
``step_batch`` signature follows the plangym contract
(https://github.com/FragileTech/plangym).
"""
import ctypes
from typing import Optional

import numpy as np

from ._rom import ROM
from .nes_env import _LIB, SCREEN_HEIGHT, SCREEN_WIDTH

# setup the argument and return types for the pool API
_LIB.InitializePool.argtypes = [
    ctypes.c_wchar_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.InitializePool.restype = ctypes.c_void_p
_LIB.ClosePool.argtypes = [ctypes.c_void_p]
_LIB.ClosePool.restype = None
_LIB.PoolNumEnvs.argtypes = [ctypes.c_void_p]
_LIB.PoolNumEnvs.restype = ctypes.c_int
_LIB.PoolMaxStateSize.argtypes = [ctypes.c_void_p]
_LIB.PoolMaxStateSize.restype = ctypes.c_size_t
_LIB.PoolResetAll.argtypes = [ctypes.c_void_p]
_LIB.PoolResetAll.restype = None
_LIB.PoolReset.argtypes = [ctypes.c_void_p, ctypes.c_int]
_LIB.PoolReset.restype = None
_LIB.PoolStepBatch.argtypes = [
    ctypes.c_void_p,  # pool
    ctypes.c_void_p,  # actions (uint8[n])
    ctypes.c_void_p,  # dt (int32[n])
    ctypes.c_int,     # frameskip
    ctypes.c_void_p,  # states_in (or None)
    ctypes.c_void_p,  # states_out (or None)
    ctypes.c_void_p,  # screens_out (or None)
    ctypes.c_void_p,  # ram_out (or None)
]
_LIB.PoolStepBatch.restype = None
_LIB.PoolLoadStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.PoolLoadStates.restype = None
_LIB.PoolDumpStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.PoolDumpStates.restype = None
_LIB.PoolScreens.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.PoolScreens.restype = None
_LIB.PoolRAM.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_LIB.PoolRAM.restype = None
_LIB.MaxStateSize.argtypes = [ctypes.c_void_p]
_LIB.MaxStateSize.restype = ctypes.c_size_t


class NESVecEnv:
    """A batch of NES emulators stepped in parallel by native C++ threads."""

    def __init__(self, rom_path, num_envs, n_workers=0, frameskip=1,
                 screen_in_state=False):
        """
        Create a new vectorized NES environment.

        Args:
            rom_path (str): the path to the ROM every emulator runs
            num_envs (int): the number of emulator instances in the batch
            n_workers (int): the number of C++ worker threads; 0 selects the
                hardware concurrency (clamped to num_envs)
            frameskip (int): emulator frames per dt unit in step_batch
            screen_in_state (bool): whether states carry the screen frame
                buffer (the legacy ~250KB format). Off by default: states
                are ~4.5KB and screens are redrawn after stepping, so
                step_batch observations are unaffected; only reading the
                screen after set_states without stepping shows stale pixels.
                States are only compatible between environments that agree
                on this setting.

        """
        # run the same ROM validation as NESEnv
        rom = ROM(rom_path)
        if rom.prg_rom_size == 0:
            raise ValueError("ROM has no PRG-ROM banks.")
        if rom.has_trainer:
            raise ValueError("ROM has trainer. trainer is not supported.")
        _ = rom.prg_rom
        _ = rom.chr_rom
        if rom.is_pal:
            raise ValueError("ROM is PAL. PAL is not supported.")
        elif rom.mapper not in {0, 1, 2, 3}:
            msg = "ROM has an unsupported mapper number {}."
            raise ValueError(msg.format(rom.mapper))
        self._rom_path = rom_path
        self.num_envs = int(num_envs)
        self.frameskip = int(frameskip)
        self._pool = _LIB.InitializePool(
            rom_path, self.num_envs, int(n_workers), int(screen_in_state))
        # the fixed state row stride; constant for the life of the pool,
        # unlike StateSize which varies by up to 8 bytes with PPU timing
        self.state_size = _LIB.PoolMaxStateSize(self._pool)
        n = self.num_envs
        # zeroed once so the padding bytes past each dump are reproducible
        self._states = np.zeros((n, self.state_size), dtype=np.uint8)
        self.screens = np.empty((n, SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        self.rams = np.empty((n, 0x800), dtype=np.uint8)
        self._dt = np.empty(n, dtype=np.int32)
        self._actions = np.empty(n, dtype=np.uint8)

    def step_batch(self, actions, states=None, dt=1, return_state=None):
        """
        Step every emulator in parallel with one native call.

        Args:
            actions: length num_envs sequence of controller bitmaps
            states: optional length num_envs sequence of state arrays to load
                before stepping (each from get_state / get_states)
            dt (int or sequence): per-element multiplier; each emulator runs
                dt[i] * frameskip frames with the action held down
            return_state (bool): whether to prepend the new states to the
                return tuple; defaults to True when states is given

        Returns:
            a tuple of lists, each of length num_envs:
            (new_states,) if return_state, then
            (observations, rewards, terminals, truncateds, infos)

        """
        n = self.num_envs
        if return_state is None:
            return_state = states is not None
        self._actions[:] = actions
        self._dt[:] = dt
        if states is not None:
            for i, state in enumerate(states):
                state = np.ascontiguousarray(state).view(np.uint8)
                self._states[i, : state.size] = state
            states_in = self._states.ctypes.data
        else:
            states_in = None
        states_out = self._states.ctypes.data if return_state else None
        _LIB.PoolStepBatch(
            self._pool,
            self._actions.ctypes.data,
            self._dt.ctypes.data,
            self.frameskip,
            states_in,
            states_out,
            self.screens.ctypes.data,
            self.rams.ctypes.data,
        )
        rewards = self._batch_reward(self.rams)
        terminals = self._batch_done(self.rams)
        infos = self._batch_info(self.rams)
        observations = [self.screens[i].copy() for i in range(n)]
        data = (
            observations,
            list(np.asarray(rewards, dtype=np.float32)),
            list(np.asarray(terminals, dtype=bool)),
            [False] * n,
            infos,
        )
        if return_state:
            new_states = [self._states[i].copy() for i in range(n)]
            return (new_states,) + data
        return data

    def _batch_reward(self, rams):
        """Return a length num_envs array of rewards from the RAM batch."""
        return np.zeros(self.num_envs, dtype=np.float32)

    def _batch_done(self, rams):
        """Return a length num_envs boolean array of terminal flags."""
        return np.zeros(self.num_envs, dtype=bool)

    def _batch_info(self, rams):
        """Return a length num_envs list of info dicts from the RAM batch."""
        return [{} for _ in range(self.num_envs)]

    def reset(self):
        """Reset every emulator and return the batch of screens."""
        _LIB.PoolResetAll(self._pool)
        _LIB.PoolScreens(self._pool, self.screens.ctypes.data)
        _LIB.PoolRAM(self._pool, self.rams.ctypes.data)
        return self.screens.copy()

    def get_states(self, states: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dump every emulator's state into a (num_envs, state_size) array.

        Args:
            states: an optional preallocated uint8 output array

        Returns:
            the (num_envs, state_size) batch of states

        """
        if states is None:
            states = np.zeros((self.num_envs, self.state_size), dtype=np.uint8)
        _LIB.PoolDumpStates(self._pool, states.ctypes.data)
        return states

    def set_states(self, states) -> None:
        """
        Load a (num_envs, state_size) batch of states into the emulators.

        Args:
            states: a sequence of num_envs state arrays, or a 2D batch array

        """
        for i, state in enumerate(states):
            state = np.ascontiguousarray(state).view(np.uint8)
            self._states[i, : state.size] = state
        _LIB.PoolLoadStates(self._pool, self._states.ctypes.data)

    def close(self):
        """Close the pool and release the C++ resources."""
        if self._pool is None:
            raise ValueError("env has already been closed.")
        _LIB.ClosePool(self._pool)
        self._pool = None

    def __del__(self):
        """Close the pool if it is still open."""
        if getattr(self, "_pool", None) is not None:
            _LIB.ClosePool(self._pool)
            self._pool = None


# explicitly define the outward facing API of this module
__all__ = [NESVecEnv.__name__]
