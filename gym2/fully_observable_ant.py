"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

from string import Template

import numpy as np
import tensorflow as tf

import cythonized
from .fully_observable import FullyObservable
from .mujoco_env import MujocoEnv

class FullyObservableAnt(MujocoEnv, FullyObservable):
    """A fully-observable Ant"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('ant.xml', cythonized.python_foa_frameskip())
        self.sim.data.userdata[2] = self.sim.model.body_name2id('torso')

    # gym code
    # def _step(self, a):
    #     xposbefore = self.get_body_com("torso")[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.get_body_com("torso")[0]
    #     forward_reward = (xposafter - xposbefore)/self.dt
    #     ctrl_cost = .5 * np.square(a).sum()
    #     contact_cost = 0.5 * 1e-3 * np.sum(
    #         np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
    #     survive_reward = 1.0
    #     reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    #     state = self.state_vector()
    #     notdone = np.isfinite(state).all() \
    #         and state[2] >= 0.2 and state[2] <= 1.0
    #     done = not notdone
    #     ob = self._get_obs()
    #     return ob, reward, done, dict(
    #         reward_forward=forward_reward,
    #         reward_ctrl=-ctrl_cost,
    #         reward_contact=-contact_cost,
    #         reward_survive=survive_reward)

    def tf_reward(self, state, action, next_state):
        xposbefore = state[:, 0]
        xposafter = next_state[:, 0]
        num_coords = 3
        cfrc_size = self.sim.model.nbody * num_coords
        trunc_cfrc_ext = tf.clip_by_value(next_state[:, -_CFRC_SIZE:], -1, 1)

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * tf.reduce_sum(tf.square(action))
        contact_cost = 0.5 * 1e-3 * tf.reduce_sum(tf.square(trunc_cfrc_ext))
        survive_reward = 1.0
        return forward_reward - ctrl_cost - contact_cost + survive_reward

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
    #         size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.sim.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * .1
        obs, _, _, = self.get_obs()
        return obs

    def set_state_from_ob(self, ob):
        self.reset()
        split = 1 + self.init_qpos.size
        end = split + self.init_qvel.size
        qpos = ob[1:split].reshape(self.init_qpos.shape)
        qvel = ob[split:end].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)

    # implemented in cython in cy_fully_observable_ant.pyx

    def get_obs(self):
        return cythonized.python_foa_get_obs(self.sim)

    def c_get_obs_fn(self):
        return cythonized.python_foa_c_get_obs_fn()

    def obs_shape(self):
        return cythonized.python_foa_obs_shape(self.sim.model)

    def prestep_callback(self, actions):
        return cythonized.python_foa_prestep(self.sim, actions)

    def c_prestep_callback_fn(self):
        return cythonized.python_foa_c_prestep_fn()

    def poststep_callback(self):
        return cythonized.python_foa_poststep(self.sim)

    def c_poststep_callback_fn(self):
        return cythonized.python_foa_c_poststep_fn()
