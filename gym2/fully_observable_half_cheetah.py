"""
The usual gym environments might withhold some items from the observation which
make it impossible to compute the reward from the observation alone.

This file contains amended environments with sufficient information
to compute reward from observations to make the settings a proper
fully-observable MDPs. Besides adding the additional dimensions to the
observations space, the environments are equivalent to the OpenAI gym
versions at commit f724a720061d76dc315b4737f19a33949cf7b687.

This file contains a fully-observable analogue of the HalfCheetah gym
environment.
"""

from string import Template

import numpy as np
import tensorflow as tf
from mujoco_py.builder import build_callback_fn

from .fully_observable import FullyObservable
from .mujoco_env import MujocoEnv

_FRAMESKIP = 5

# before taking any steps, store the initial x position
_PRESTEP = """
void fun(const mjModel* m, mjData* d) {
  d->userdata[0] = d->qpos[0];
}
"""

# after taking all the steps, store the resulting reward in userdata[0]
# and whether we're done in userdata[1]
_POSTSTEP = Template("""
void fun(const mjModel* m, mjData* d) {
  double x_before = d->userdata[0];
  double x_after = d->qpos[0];
  double control_magnitude = 0;
  double dt = m->opt.timestep * $FRAMESKIP;
  for (int i = 0; i < m->nu; ++i) {
    control_magnitude += d->ctrl[i] * d->ctrl[i];
  }
  d->userdata[0] = (x_after - x_before) / dt - 0.1 * control_magnitude;
  d->userdata[1] = 0;
}
""").substitute(FRAMESKIP=_FRAMESKIP)

_PRESTEP_CALLBACK_FN = build_callback_fn(_PRESTEP)
_POSTSTEP_CALLBACK_FN = build_callback_fn(_POSTSTEP)


class FullyObservableHalfCheetah(MujocoEnv, FullyObservable):
    """A fully-observable HalfCheetah"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('half_cheetah.xml', _FRAMESKIP,
                         prestep_callback_ptr=_PRESTEP_CALLBACK_FN,
                         poststep_callback_ptr=_POSTSTEP_CALLBACK_FN)

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
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[1:],
    #         self.model.data.qvel.flat,
    #     ])

    def get_obs(self, out_obs):
        pos_size = self.sim.data.qpos.size
        out_obs[:pos_size] = self.sim.data.qpos
        out_obs[pos_size:] = self.sim.data.qvel
        return out_obs

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
    #         size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        obs = np.empty(self._obs_shape())
        self.get_obs(obs)
        return obs

    def set_state_from_ob(self, ob):
        self.reset()
        split = self.init_qpos.size
        qpos = ob[:split].reshape(self.init_qpos.shape)
        qvel = ob[split:].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)

    def _obs_shape(self):
        pos_size = self.sim.data.qpos.size
        vel_size = self.sim.data.qpos.size
        return (pos_size + vel_size,)
