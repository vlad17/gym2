"""
A vectorized environment can be viewed formally as a Cartesian product of
identical MDPs. It lets us evaluate several trajectories in parallel.
"""

import os
import numpy as np

import gym


class VectorEnv(gym.Env):
    """
    Abstract base class for vectorized environments. These environments
    are structured identically to gym environments but they omit the
    following information:

    * The "info" dictionary, which is the last return value in the tuple
      from step() will be empty.
    * Rendering is unsupported.
    * Environment wrappers are not supported (the environments being
      vectorized should be wrapped instead).

    However, vectorized envs offer a couple additional useful pieces
    of functionality.

    set_state_from_obs - does what the name says, as long as the
    underlying env has a set_state_from_ob method (this presumes a
    fully observable MDP).

    Since some environments may terminate early while others are
    iterating, vectorized environments automatically mask out done
    environments from taking further actions until the next reset().
    The caller is still responsible for tracking which environment
    indices are active, since there is no guarantee on the returned
    values for masked environments.

    Moreover, vectorized environments handle being passed in arguments
    with smaller rank gracefully. E.g., if a vectorized environment has
    rank n (n simultaneous vectorized environments), it will accept
    actions of length m <= n and only simulate the first m environments.
    """

    def set_state_from_ob(self, obs):
        """
        Set the state for each environment with the given observations.
        Requires underlying environments have the set_state_from_ob method.
        """
        raise NotImplementedError

    def _seed(self, seed=None):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _close(self):
        raise NotImplementedError

    def _seed_uncorr(self, n):
        rand_bytes = os.urandom(n * 4)
        rand_uints = np.frombuffer(rand_bytes, dtype=np.uint32)
        self.seed([int(x) for x in rand_uints])
