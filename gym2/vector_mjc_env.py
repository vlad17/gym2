"""
This class vectorizes mulitple gym2 mujoco environments across several CPUs.
"""

import numpy as np

from .cythonized import MjSimPool
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
        frame_skips = set(env.frame_skip for env in self._envs)
        assert len(frame_skips) == 1, frame_skips
        env = self._envs[0]

        obs_copy_fns = [env.c_get_obs_fn() for env in self._envs]
        prestep_callbacks = [env.c_prestep_callback_fn() for env in self._envs]
        poststep_callbacks = [env.c_poststep_callback_fn() for env in self._envs]
        self._pool = MjSimPool([env.sim for env in self._envs],
                               frame_skips.pop(),
                               obs_copy_fns,
                               prestep_callbacks,
                               poststep_callbacks)
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
        for env, env_seed in zip(self._envs, seed):
            env.seed(env_seed)
        return seed

    def _reset(self):
        self._mask[:] = 1
        return np.asarray([env.reset() for env in self._envs])

    def _step(self, action):
        m = len(action)
        assert m <= len(self._envs), (m, len(self._envs))
        obs = np.empty((m,) + self.observation_space.shape)
        rews = np.empty((m,))
        dones = np.empty((m,), dtype=np.uint8)
        infos = [{}] * m
        acs = np.asarray(action)
        mask_char = self._mask.astype(np.uint8, copy=False)
        self._pool.step(np.asarray(acs, dtype=float), obs, rews, dones,
                        nsims=m, mask=mask_char)
        return obs, rews, dones.astype(bool, copy=False), infos

    def _close(self):
        for env in self._envs:
            env.close()

    def multi_step(self, acs_hna):
        h, m = acs_hna.shape[:2]
        assert m <= self.n, (m, self.n)
        obs = np.empty((h, m,) + self.observation_space.shape)
        rews = np.empty((h, m,))
        dones = np.empty((h, m,), dtype=bool)
        for i in range(h):
            obs[i], rews[i], dones[i], _ = self.step(acs_hna[i])
        return obs, rews, dones
