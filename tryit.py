from contextlib import closing
from functools import partial
import sys
import time

import gym
import numpy as np

gym.undo_logger_setup()


def _usage():
    print('Usage: python tryit.py envname {render,bench} {old,new}',
          file=sys.stderr)
    sys.exit(1)


def _use_old_gym():
    assert sys.argv[3] in ['old', 'new']
    return sys.argv[3] == 'old'


def _env(envname):
    if _use_old_gym():
        from gym2 import analogues
        envs = {
            'hc': analogues.GymFullyObservableHalfCheetah,
            'ant': analogues.GymFullyObservableAnt
        }
    else:
        import gym2
        envs = {
            'hc': gym2.FullyObservableHalfCheetah,
            'ant': gym2.FullyObservableAnt
        }
    if envname not in envs:
        _usage()
    return envs[envname]


def print_is_gym2():
    if _use_old_gym():
        print('using old gym')
    else:
        print('using gym2')


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
    print_is_gym2()
    with closing(envclass()) as env:
        if not _use_old_gym():
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

        print_is_gym2()
        print('runtime for ~65K steps {:.4g}'.format(end - start))
        if _use_old_gym():
            from gym2 import ParallelGymVenv
            par_env = ParallelGymVenv
        else:
            from gym2 import VectorMJCEnv
            par_env = VectorMJCEnv
        import multiprocessing as mp
        cpu = mp.cpu_count()
        # https://stackoverflow.com/questions/14267555
        # round up to power of 2
        cpu = 1 << (cpu - 1).bit_length()
        for par in [1, 2, 4, cpu, 512]:
            with closing(par_env(par, envclass)) as venv:
                obs = venv.reset()
                venv.step(acs[:par])  # warmup
                start = time.time()
                for ac in acs.reshape(-1, par, *acs.shape[1:]):
                    _, _, done, _ = venv.step(ac)
                    if np.all(done):
                        venv.reset()
                end = time.time()
                steptime = end - start

                start = time.time()
                for _ in range(len(acs) // par):
                    venv.set_state_from_ob(obs)
                end = time.time()

            print('   parallelized over {} envs {:.4g}'
                  '  (set_state_from_ob {:.4g})'.format(
                      par, steptime, end - start))


def _main():
    if len(sys.argv) != 4:
        _usage()

    envname, option = sys.argv[1:3]
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
