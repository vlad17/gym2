from contextlib import closing
import sys

import gym
import numpy as np

from gym2 import FullyObservableHalfCheetah

import mujoco_py

print(mujoco_py.get_version())


def _usage():
    print('Usage: python tryit.py envname [render|bench]', file=sys.stderr)
    sys.exit(1)


def _random_actions(env, n):
    space = env.action_space
    acs = np.random.uniform(size=(n,) + space.shape)
    if all(np.isfinite(space.low)) and all(np.isfinite(space.high)):
        acs *= space.high - space.low
        acs += space.low
    else:
        acs *= 2
        acs -= 1
    return acs


def _render(envclass):
    horiz = 100
    with closing(envclass()) as env:
        limited_env = gym.wrappers.TimeLimit(env, max_episode_steps=horiz)
        with closing(gym.wrappers.Monitor(
                limited_env, 'render', force=True)) as rendered:
            rendered.reset()
            for ac in _random_actions(rendered, horiz):
                rendered.render()
                rendered.step(ac)


def _bench(envclass):
    pass


def _main():
    if len(sys.argv) != 3:
        _usage()

    envname, option = sys.argv[1:]
    should_render = {
        'render': True,
        'bench': False
    }
    envs = {
        'hc': FullyObservableHalfCheetah
    }
    if option not in should_render or envname not in envs:
        _usage()

    np.random.seed(1234)

    if should_render:
        _render(envs[envname])
    else:
        _bench(envs[envname])


if __name__ == '__main__':
    _main()
