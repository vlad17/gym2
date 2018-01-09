"""
This class vectorizes mulitple mujoco environments across several CPUs.
"""

from mujoco_py import MjSimPool
import numpy as np

from .vector_env import VectorEnv


class VectorMJCEnv(VectorEnv):
    """
    Accepts a no-argument lambda that generates the scalar environments,
    which should be instances of MujocoEnv, that should be vectorized
    together.

    Vectorizes n environments.

    Does NOT allow for rendering. This is a gym.Env over the MDP which
    is a Cartesian product of its contained MDPs, so all actions should
    have an additional first axis where each element along that axis
    is an action for each scalar subenvironment. All observations,
    rewards, and "done states" are correspondingly vectorized.

    For more information, see VectorEnv. It contains more information
    on what happens when some of the environments are done while others
    are active.
    """

    def __init__(self, n, scalar_env_gen):
        assert n > 0, n
        self._envs = [scalar_env_gen() for _ in range(n)]
        self._pool = MjSimPool([env.sim for env in self._envs])

        env = self._envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
        self.n = len(self._envs)
        self._mask = np.ones(self.n, dtype=bool)
        self._seed_uncorr(self.n)

    def set_state_from_ob(self, obs):
        for ob, env in zip(obs, self._envs):
            env.set_state_from_ob(ob)

    def _seed(self, seed=None):
        seeds = []
        for env, env_seed in zip(self._envs, seed):
            seeds.append(env.seed(env_seed))
        return seeds

    def _reset(self):
        self._mask[:] = True
        return np.asarray([env.reset() for env in self._envs])

    def _step(self, action):
        # TODO add mask / m handling (masking just to prevent errors)
        m = len(action)
        assert m <= len(self._envs), (m, len(self._envs))
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=bool)
        infos = [{}] * m
        for env, ac in zip(self._envs, action):
            env.sim.data.ctrl[:] = ac
        self._pool.step()
        for i, env in enumerate(self._envs[:m]):
            env.get_obs(obs[i])
            rews[i] = env.sim.data.userdata[0]
            dones[i] = env.sim.data.userdata[1] > 0
        return obs, rews, dones, infos

    def _close(self):
        for env in self._envs:
            env.close()
