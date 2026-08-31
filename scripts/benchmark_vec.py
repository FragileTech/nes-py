"""Benchmark the native batch NESVecEnv against a serial NESEnv loop.

The serial baseline emulates what planning code does today with one env in
Python: for every batch element, set_state -> step dt frames -> get_state.
The vectorized path does the same work in a single native call.
"""
import argparse
import os
import time

import numpy as np

from nes_py import NESEnv, NESVecEnv


ROM_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nes_py", "tests", "games", "super-mario-bros-1.nes",
)


def bench_serial(state_0, batch_size, dt, iters, with_states):
    """Time the classic Python loop: per element set_state/step/get_state."""
    env = NESEnv(ROM_PATH)
    env.reset()
    states = [state_0.copy() for _ in range(batch_size)]
    start = time.perf_counter()
    for _ in range(iters):
        new_states = []
        for i in range(batch_size):
            if with_states:
                env.set_state(states[i])
            for _ in range(dt):
                env._frame_advance(8)
            if with_states:
                new_states.append(env.get_state())
            _ = env.screen
            _ = env.ram
    elapsed = time.perf_counter() - start
    env.close()
    return elapsed


def bench_vec(state_0, batch_size, n_workers, dt, iters, with_states):
    """Time the native batch call."""
    vec = NESVecEnv(ROM_PATH, num_envs=batch_size, n_workers=n_workers)
    vec.reset()
    states = [state_0.copy() for _ in range(batch_size)] if with_states else None
    actions = [8] * batch_size
    # warmup
    vec.step_batch(actions, states=states, dt=dt, return_state=with_states)
    start = time.perf_counter()
    for _ in range(iters):
        vec.step_batch(actions, states=states, dt=dt, return_state=with_states)
    elapsed = time.perf_counter() - start
    vec.close()
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dt", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--workers", type=int, nargs="+", default=None)
    args = parser.parse_args()
    n_cpu = os.cpu_count()
    workers = args.workers or sorted({1, 4, n_cpu})

    env = NESEnv(ROM_PATH)
    env.reset()
    state_0 = env.get_state()
    env.close()

    print("cpus: {}  dt: {}  iters: {}".format(n_cpu, args.dt, args.iters))
    header = "{:>6} {:>8} {:>7} {:>12} {:>12} {:>8}".format(
        "batch", "workers", "states", "serial fps", "vec fps", "speedup")
    for with_states in (True, False):
        print("\n--- {} state set/get per step ---".format(
            "WITH" if with_states else "WITHOUT"))
        print(header)
        for batch_size in args.sizes:
            frames = args.iters * batch_size * args.dt
            t_serial = bench_serial(
                state_0, batch_size, args.dt, args.iters, with_states)
            for n_workers in workers:
                if n_workers > batch_size:
                    continue
                t_vec = bench_vec(
                    state_0, batch_size, n_workers, args.dt,
                    args.iters, with_states)
                print("{:>6} {:>8} {:>7} {:>12.0f} {:>12.0f} {:>7.1f}x".format(
                    batch_size, n_workers, str(with_states),
                    frames / t_serial, frames / t_vec, t_serial / t_vec))


if __name__ == "__main__":
    main()
