import pytest
import numpy as np

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


@pytest.fixture(scope="function")
def env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    _ = env.reset()
    return env


def _step_env(env, steps=50):
    done = False
    for _ in range(steps):
        if done:
            break
        *_, done, info = env.step(env.action_space.sample())
    return info


def test_get_state(env):
    env.reset()
    state_0 = env.state.copy()
    _step_env(env)
    assert not (state_0 == env.state).all()
    env.reset()
    assert (state_0 == env.state).all()


def test_set_state(env):
    env.reset()
    state_0 = env.state.copy()
    _step_env(env)
    state_1 = env.state.copy()
    assert not (state_0 == state_1).all()
    for _ in range(2):
        env.step(2)
    state_3 = env.state.copy()
    env.set_state(state_0.copy())
    assert (state_0 == env.state).all()
    env.set_state(state_1.copy())
    assert (state_1 == env.state).all()
    for _ in range(2):
        env.step(2)
    assert (state_3 == env.state).all()
