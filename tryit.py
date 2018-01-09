from contextlib import closing
import sys
import time

import gym
import numpy as np
import mujoco_py


def _usage():
    print('Usage: python tryit.py envname [render|bench]', file=sys.stderr)
    sys.exit(1)


def _mjpv():
    if hasattr(mujoco_py, 'get_version'):
        return mujoco_py.get_version()
    return '0.5.7'  # yeah, just assume


def _env(envname):
    if _mjpv() == '0.5.7':
        envs = {
            'hc': lambda: gym.make('HalfCheetah-v1')
        }
    else:
        msg = 'you should be using vlad17/mujoco_py branch pre-post-callbacks'
        assert _mjpv() == '1.50.1.99999', msg
        from gym2 import FullyObservableHalfCheetah
        envs = {
            'hc': FullyObservableHalfCheetah
        }
    if envname not in envs:
        _usage()
    return envs[envname]


def is_gym2():
    if _mjpv() == '0.5.7':
        print('mujoco_py version {}, using old gym'.format(_mjpv()))
        return False
    print('mujoco_py version {}, using gym2'.format(_mjpv()))
    return True


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


def _render(envname):
    horiz = 100
    envclass = _env(envname)
    with closing(envclass()) as env:
        if is_gym2():
            env = gym.wrappers.TimeLimit(env, max_episode_steps=horiz)
        with closing(gym.wrappers.Monitor(
                env, 'render', force=True)) as rendered:
            rendered.reset()
            for ac in _random_actions(rendered, horiz):
                rendered.render()
                rendered.step(ac)


def _bench(envname):
    envclass = _env(envname)
    with closing(envclass()) as env:
        env.reset()
        start = time.time()
        acs = _random_actions(env, 2 ** 16)
        for ac in acs:
            _, _, done, _ = env.step(ac)
            if done:
                env.reset()
        end = time.time()

        is_gym2_bool = is_gym2()
        print('runtime for ~65K steps {:.4g}'.format(end - start))
        if is_gym2_bool:
            from gym2.vector_mjc_env import VectorMJCEnv

            for par in [64]:  # [2, 4, 8, 16, 32, 64, 128, 256, 512]:
                with closing(VectorMJCEnv(par, envclass)) as venv:
                    start = time.time()
                    for ac in acs.reshape(-1, par, *acs.shape[1:]):
                        _, _, done, _ = venv.step(ac)
                        if np.all(done):
                            venv.reset()
                    end = time.time()
                print('   parallelized over {} envs {:.4g}'.format(
                    par, end - start))


def _main():
    if len(sys.argv) != 3:
        _usage()

    envname, option = sys.argv[1:]
    should_render = {
        'render': True,
        'bench': False
    }
    np.random.seed(1234)

    if should_render[option]:
        _render(envname)
    else:
        _bench(envname)


if __name__ == '__main__':
    _main()
