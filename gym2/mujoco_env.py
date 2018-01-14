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

import cythonized
from .mujoco.generated import const


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
        model = cythonized.load_model_from_path(fullpath)
        self.sim = cythonized.MjSim(
           model, nsubsteps=self.frame_skip, **mjsim_kwargs)
        self._model_ptr = cythonized.mj_model_ptr(self.sim.model)
        self._data_ptr = cythonized.mj_data_ptr(self.sim.data)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.obs_shape = self._obs_shape()

        bounds = self.sim.model.actuator_ctrlrange.copy()
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

    def _get_obs(self, out_obs, out_rew):
        """
        Provide an observation in the observation space given the current data,
        available in self.sim, writing into the given output array.

        Same follows for the output reward array, a size-1 array of a float.

        Return a boolean whether done.
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
            self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        new_state = cythonized.MjSimState(old_state.time, qpos, qvel,
                                          old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip

    def _step(self, ctrl):
        self.sim.data.ctrl[:] = ctrl
        self.sim.step()
        obs = np.empty(self.obs_shape)
        reward = np.empty((1,))
        done = np.empty((1,), dtype=np.uint8)
        self._get_obs(
            obs, reward, done, self._model_ptr, self._data_ptr)
        info = {}
        return obs, reward[0], done[0], info

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
            ctx.cam.distance = self.sim.model.stat.extent * 0.5
            ctx.cam.type = const.CAMERA_TRACKING
            ctx.cam.trackbodyid = 0
