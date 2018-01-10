"""
This file was made in the style of fully_observable_half_cheetah.py,
mimicking the analogous gym environment.
"""

from string import Template

import numpy as np
import tensorflow as tf
import six

from .builder import build_callback_fn
from .fully_observable import FullyObservable
from .mujoco_env import MujocoEnv

_CFRC_SIZE = 84
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
# include <math.h>
void fun(const mjModel* m, mjData* d) {
  double x_before = d->userdata[0];
  double x_after = d->qpos[0];

  double truncated_cfrc_magnitude = 0;
  for (int i = 0; i < $CFRC_SIZE; ++i) {
    truncated_cfrc_magnitude += d->cfrc_ext[i] * d->cfrc_ext[i];
  }

  double control_magnitude = 0;
  double dt = m->opt.timestep * $FRAMESKIP;
  for (int i = 0; i < m->nu; ++i) {
    control_magnitude += d->ctrl[i] * d->ctrl[i];
  }
  d->userdata[0] = (x_after - x_before) / dt;
  d->userdata[0] -= 0.5 * control_magnitude;
  d->userdata[0] -= 0.5 * 1.0e-3 * truncated_cfrc_magnitude;
  d->userdata[0] += 1.0;

  d->userdata[1] = 0;
  for (int i = 0; i < m->nq; ++i) {
    if (!isfinite(d->qpos[i])) {
      d->userdata[1] = 1;
    }
  }
  for (int i = 0; i < m->nv; ++i) {
    if (!isfinite(d->qvel[i])) {
      d->userdata[1] = 1;
    }
  }
  if (d->qpos[2] < 0.2 || d->qpos[2] > 1) {
    d->userdata[1] = 1;
  }
}
""").substitute(FRAMESKIP=_FRAMESKIP, CFRC_SIZE=_CFRC_SIZE)

_PRESTEP_CALLBACK_FN = build_callback_fn(_PRESTEP)
_POSTSTEP_CALLBACK_FN = build_callback_fn(_POSTSTEP)


class FullyObservableAnt(MujocoEnv, FullyObservable):
    """A fully-observable Ant"""

    # gym code
    # def __init__(self):
    #     mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
    #     utils.EzPickle.__init__(self)

    def __init__(self):
        super().__init__('ant.xml', _FRAMESKIP,
                         prestep_callback_ptr=_PRESTEP_CALLBACK_FN,
                         poststep_callback_ptr=_POSTSTEP_CALLBACK_FN)

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
        trunc_cfrc_ext = tf.clip_by_value(next_state[:, -_CFRC_SIZE:], -1, 1)

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * tf.reduce_sum(tf.square(action))
        contact_cost = 0.5 * 1e-3 * tf.reduce_sum(tf.square(trunc_cfrc_ext))
        survive_reward = 1.0
        return forward_reward - ctrl_cost - contact_cost + survive_reward

    # gym code
    # def _get_obs(self):
    #     return np.concatenate([
    #         self.model.data.qpos.flat[1:],
    #         self.model.data.qvel.flat,
    #     ])

    def _obs_shape(self):
        pos_size = self.sim.data.qpos.size
        vel_size = self.sim.data.qvel.size
        cfrc_size = self.sim.data.cfrc_ext.size
        assert cfrc_size == _CFRC_SIZE, (cfrc_size, _CFRC_SIZE)
        return (1 + pos_size + vel_size + cfrc_size,)

    def get_obs(self, out_obs):
        # difference from gym: need torso x value for reward
        i = 0
        out_obs[i] = self.sim.data.get_body_xpos('torso')[0]
        i += 1
        pos_size = self.sim.data.qpos.size
        # difference from gym: need to add in qpos x value
        # so that states are re-settable
        out_obs[i:i + pos_size] = self.sim.data.qpos
        i += pos_size
        vel_size = self.sim.data.qvel.size
        out_obs[i:i + vel_size] = self.sim.data.qvel
        i += vel_size
        # difference from gym: need contact forces for reward
        out_obs[i:] = self.sim.data.cfrc_ext.ravel()

    # gym code
    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1,
    #         size=self.model.nq)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        obs = np.empty(self._obs_shape())
        self.get_obs(obs)
        return obs

    def set_state_from_ob(self, ob):
        self.reset()
        split = 1 + self.init_qpos.size
        end = split + self.init_qvel.size
        qpos = ob[1:split].reshape(self.init_qpos.shape)
        qvel = ob[split:end].reshape(self.init_qvel.shape)
        self.set_state(qpos, qvel)
