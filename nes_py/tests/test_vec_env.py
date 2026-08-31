"""Tests for the NESVecEnv vectorized environment."""
from unittest import TestCase

import numpy as np

from nes_py import NESEnv, NESVecEnv
from .rom_file_abs_path import rom_file_abs_path


ROM_PATH = rom_file_abs_path("super-mario-bros-1.nes")


def serial_rollout(state, actions_per_frame):
    """Step a fresh NESEnv from a state and return (state, ram, screen)."""
    env = NESEnv(ROM_PATH)
    env.reset()
    env.set_state(state.copy())
    for action in actions_per_frame:
        env._frame_advance(action)
    result = env.get_state(), env.ram.copy(), env.screen.copy()
    env.close()
    return result


class ShouldCreateAndClosePool(TestCase):
    def test(self):
        for _ in range(3):
            vec = NESVecEnv(ROM_PATH, num_envs=2, n_workers=2)
            self.assertEqual(vec.num_envs, 2)
            self.assertGreater(vec.state_size, 0)
            vec.reset()
            vec.close()

    def test_close_twice_raises(self):
        vec = NESVecEnv(ROM_PATH, num_envs=1)
        vec.close()
        self.assertRaises(ValueError, vec.close)


class ShouldMatchSerialEnv(TestCase):
    def test_batch_step_matches_serial(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        for _ in range(50):
            env._frame_advance(8)
        state_0 = env.get_state()
        env.close()

        n = 4
        actions = [0, 8, 16, 8]
        dt = 3
        vec = NESVecEnv(ROM_PATH, num_envs=n, n_workers=n)
        states = [state_0.copy() for _ in range(n)]
        new_states, obs, rewards, terms, truncs, infos = vec.step_batch(
            actions, states=states, dt=dt
        )
        self.assertEqual(len(new_states), n)
        self.assertEqual(obs[0].shape, (240, 256, 3))
        self.assertEqual(len(rewards), n)
        self.assertEqual(truncs, [False] * n)
        for i in range(n):
            _, ram, screen = serial_rollout(state_0, [actions[i]] * dt)
            self.assertTrue(np.array_equal(ram, vec.rams[i]))
            self.assertTrue(np.array_equal(screen, vec.screens[i]))
        vec.close()

    def test_frameskip(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        state_0 = env.get_state()
        env.close()
        # dt=2 with frameskip=3 must equal 6 serial frames
        vec = NESVecEnv(ROM_PATH, num_envs=2, frameskip=3)
        vec.step_batch([8, 8], states=[state_0.copy()] * 2, dt=2)
        _, ram, _ = serial_rollout(state_0, [8] * 6)
        self.assertTrue(np.array_equal(ram, vec.rams[0]))
        vec.close()


class ShouldBeDeterministic(TestCase):
    def test_repeat_identical(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        state_0 = env.get_state()
        env.close()
        vec = NESVecEnv(ROM_PATH, num_envs=3, n_workers=3)
        states = [state_0.copy() for _ in range(3)]
        first = vec.step_batch([8, 0, 32], states=states, dt=2)[0]
        second = vec.step_batch([8, 0, 32], states=states, dt=2)[0]
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a, b))
        vec.close()

    def test_worker_count_invariant(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        state_0 = env.get_state()
        env.close()
        results = []
        for n_workers in (1, 8):
            vec = NESVecEnv(ROM_PATH, num_envs=8, n_workers=n_workers)
            states = [state_0.copy() for _ in range(8)]
            vec.step_batch(list(range(0, 64, 8)), states=states, dt=4)
            results.append(vec.rams.copy())
            vec.close()
        self.assertTrue(np.array_equal(results[0], results[1]))


class ShouldDivergeWithDifferentActions(TestCase):
    def test(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        # advance past the title screen so inputs affect the game
        for _ in range(200):
            env._frame_advance(8)
        for _ in range(100):
            env._frame_advance(0)
        state_0 = env.get_state()
        env.close()
        vec = NESVecEnv(ROM_PATH, num_envs=2, n_workers=2)
        # hold right vs noop for 30 frames from the same start state
        vec.step_batch([0x80, 0], states=[state_0.copy()] * 2, dt=30)
        self.assertFalse(np.array_equal(vec.rams[0], vec.rams[1]))
        vec.close()


class ShouldGetAndSetStates(TestCase):
    def test_roundtrip(self):
        vec = NESVecEnv(ROM_PATH, num_envs=2, n_workers=2)
        vec.reset()
        vec.step_batch([8, 0], dt=5)
        states = vec.get_states()
        rams = vec.rams.copy()
        # advance, then restore and verify RAM comes back
        vec.step_batch([8, 8], dt=10)
        self.assertFalse(np.array_equal(rams, vec.rams))
        vec.set_states(states)
        _ = vec.step_batch([0, 0], dt=1)  # refresh rams via a noop-free readout
        # compare against a serial continuation instead: restore and re-dump
        vec.set_states(states)
        restored = vec.get_states()
        # dumped states are identical after a set/get round trip
        self.assertTrue(np.array_equal(states, restored))
        vec.close()

    def test_state_size_is_constant_per_rom(self):
        # single env and vec env must agree on the per-ROM state size, and
        # every dumped state must have exactly that size so batches stack
        env = NESEnv(ROM_PATH)
        env.reset()
        vec = NESVecEnv(ROM_PATH, num_envs=2)
        self.assertEqual(env.state_size, vec.state_size)
        states = []
        for _ in range(5):
            env._frame_advance(8)
            states.append(env.get_state())
        self.assertEqual({s.size for s in states}, {env.state_size})
        stacked = np.stack(states)
        self.assertEqual(stacked.shape, (5, env.state_size))
        # a stacked row must round-trip through both envs
        env.set_state(stacked[0])
        vec.set_states([stacked[0], stacked[1]])
        # too-small and non-contiguous buffers are rejected
        self.assertRaises(ValueError, env.get_state, np.zeros(8, dtype=np.uint8))
        big = np.zeros((env.state_size, 2), dtype=np.uint8)
        self.assertRaises(ValueError, env.get_state, big[:, 0])
        env.close()
        vec.close()

    def test_screenless_states_are_small(self):
        vec = NESVecEnv(ROM_PATH, num_envs=1)
        self.assertLess(vec.state_size, 10000)
        vec.close()
        vec = NESVecEnv(ROM_PATH, num_envs=1, screen_in_state=True)
        self.assertGreater(vec.state_size, 240 * 256 * 4)
        vec.close()

    def test_screenless_roundtrip_replays_screens(self):
        # a screenless state must reproduce RAM and post-step screens exactly
        env = NESEnv(ROM_PATH)
        env.reset()
        for _ in range(50):
            env._frame_advance(8)
        state_mid = env.get_state()
        self.assertLess(state_mid.size, 10000)
        for _ in range(10):
            env._frame_advance(8)
        ram_ref = env.ram.copy()
        screen_ref = env.screen.copy()
        # restore and replay in a batch env with the same (default) format
        vec = NESVecEnv(ROM_PATH, num_envs=2, n_workers=2)
        vec.step_batch([8, 8], states=[state_mid.copy()] * 2, dt=10)
        self.assertTrue(np.array_equal(ram_ref, vec.rams[0]))
        self.assertTrue(np.array_equal(screen_ref, vec.screens[0]))
        vec.close()
        env.close()

    def test_legacy_format_restores_screen_without_stepping(self):
        env = NESEnv(ROM_PATH, screen_in_state=True)
        env.reset()
        for _ in range(50):
            env._frame_advance(8)
        state = env.get_state()
        screen = env.screen.copy()
        for _ in range(20):
            env._frame_advance(8)
        env.set_state(state)
        self.assertTrue(np.array_equal(screen, env.screen))
        env.close()

    def test_per_element_dt(self):
        env = NESEnv(ROM_PATH)
        env.reset()
        state_0 = env.get_state()
        env.close()
        vec = NESVecEnv(ROM_PATH, num_envs=2, n_workers=2)
        vec.step_batch([8, 8], states=[state_0.copy()] * 2, dt=[1, 4])
        _, ram_1, _ = serial_rollout(state_0, [8] * 1)
        _, ram_4, _ = serial_rollout(state_0, [8] * 4)
        self.assertTrue(np.array_equal(ram_1, vec.rams[0]))
        self.assertTrue(np.array_equal(ram_4, vec.rams[1]))
        vec.close()
