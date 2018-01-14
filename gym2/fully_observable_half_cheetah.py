"""
The usual gym environments might withhold some items from the observation which
make it impossible to compute the reward from the observation alone.

This file contains amended environments with sufficient information
to compute reward from observations to make the settings a proper
fully-observable MDPs. Besides adding the additional dimensions to the
observations space, the environments are equivalent to the OpenAI gym
versions at commit f724a720061d76dc315b4737f19a33949cf7b687.

This file contains a fully-observable analogue of the HalfCheetah gym
environment. Note that part of this file's
"""

from string import Template

import tensorflow as tf

import cythonized
from .fully_observable import FullyObservable
from .mujoco_env import MujocoEnv

class FullyObservableHalfCheetah(MujocoEnv, FullyObservable):
    """A fully-observable HalfCheetah"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('half_cheetah.xml', cythonized.python_fohc_frameskip())

    # gym code
    # def _step(self, action):
    #     xposbefore = self.model.data.qpos[0, 0]
    #     self.do_simulation(action, self.frame_skip)
    #     xposafter = self.model.data.qpos[0, 0]
    #     ob = self._get_obs()
    #     reward_ctrl = - 0.1 * np.square(action).sum()
    #     reward_run = (xposafter - xposbefore)/self.dt
    #     reward = reward_ctrl + reward_run
    #     done = False
    #     return ob, reward, done, dict(reward_run=reward_run,
    #         reward_ctrl=reward_ctrl)

    def tf_reward(self, state, action, next_state):
        reward = tf.zeros([tf.shape(state)[0]])
        ac_reg = tf.reduce_sum(action * action, axis=1)
        ac_reg *= 0.1
        reward -= ac_reg
        reward += (next_state[:, 0] - state[:, 0]) / self.unwrapped.dt
        return reward

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
    #         size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.sim.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .1
        self.set_state(qpos, qvel)
        obs, _, _ = self.get_obs()
        return obs

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)

    # implemented in cython in cy_fully_observable_half_cheetah.pyx

    def get_obs(self):
        return cythonized.python_fohc_get_obs(self.sim)

    def c_get_obs_fn(self):
        return cythonized.python_fohc_c_get_obs_fn()

    def obs_shape(self):
        return cythonized.python_fohc_obs_shape(self.sim.model)

    def prestep_callback(self, actions):
        return cythonized.python_fohc_prestep(self.sim, actions)

    def c_prestep_callback_fn(self):
        return cythonized.python_fohc_c_prestep_fn()

    def poststep_callback(self):
        return cythonized.python_fohc_poststep(self.sim)

    def c_poststep_callback_fn(self):
        return cythonized.python_fohc_c_poststep_fn()
