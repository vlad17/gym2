"""
MuJoCo 1.50 compatible MujocoEnv.
Adapted from https://github.com/openai/gym/pull/767.
"""

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import mujoco_py


class MujocoEnv(gym.Env):
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, **mjsim_kwargs):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(
            self.model, nsubsteps=self.frame_skip, **mjsim_kwargs)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.obs_shape = self._obs_shape()

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf * np.ones(self.obs_shape)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def _obs_shape(self):
        """
        The shape of the observation array (for computing
        self.observation_space). The MJC model will have been loaded
        by the time this is called.
        """
        raise NotImplementedError

    def get_obs(self, out_obs):
        """
        Provide an observation in the observation space given the current data,
        available in self.sim, writing into the given output array.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (
            self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def _step(self, ctrl):
        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        obs = np.empty(self.obs_shape)
        self.get_obs(obs)
        reward = self.sim.data.userdata[0]
        done = self.sim.data.userdata[1] > 0
        info = {}
        return obs, reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer = None
            return

        if mode == 'rgb_array':
            width, height = 640, 480
            self._setup_render()
            data = self.sim.render(width, height)
            return data.reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            # avoid literally all GL pain by using off-screen renders
            pass

    def _setup_render(self):
        if self.sim._render_context_offscreen is None:
            self.sim.render(640, 480)
            assert self.sim._render_context_offscreen is not None
            ctx = self.sim._render_context_offscreen
            ctx.cam.distance = self.model.stat.extent * 0.5
            ctx.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
            ctx.cam.trackbodyid = 0
