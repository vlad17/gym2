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
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        bounds = self.sim.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf * np.ones(self.obs_shape())
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.seed()

    def seed(self, seed=None):
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

    def obs_shape(self):
        """
        The shape of the observation array (for computing
        self.observation_space). The MJC model will have been loaded
        by the time this is called.
        """
        raise NotImplementedError

    def prestep_callback(self, actions):
        """
        Called before the MuJoCo simulator is stepped.
        Given an action array, it's the subclass responsibility here
        to set the sim.data.ctrl appropriately for simulation given the actions

        Any additional pre-step information should be saved
        at this point.
        """
        raise NotImplementedError

    def c_prestep_callback_fn(self):
        """
        Returns a pointer to the C function of type:

        void c_prestep_callback(
            mjModel* model, mjData* data, double[:] actions) nogil

        The C equivalent of prestep_callback(). Return 0 if none.
        """
        raise NotImplementedError

    def poststep_callback(self):
        """
        Called after the MuJoCo simulator is stepped.
        All post-processing after stepping should be performed here.
        """
        raise NotImplementedError

    def c_poststep_callback_fn(self):
        """
        Returns a pointer to the C function of type:

        void c_poststep_callback(mjModel* model, mjData* data) nogil

        The C equivalent of poststep_callback(). Return 0 if none.
        """
        raise NotImplementedError

    def get_obs(self):
        """
        Should be called after successive step() evaluations.

        Provides a tuple of (current observation, reward, done).
        """
        raise NotImplementedError

    def c_get_obs_fn(self):
        """
        Returns pointer to C function of type:

        void c_get_obs(
            double[:] out_obs, double* out_reward,
            np.uint8_t* out_done, mjModel* model, mjData* data) nogil

        Should fill in the observations appropriately. Corresponds to
        get_obs(self).
        """
        raise NotImplementedError

    def c_set_state_fn(self):
        """
        Return a pointer to a function as an int with type:

        void set_state(mjModel* model, mjData* data, double[:] newstate) nogil

        where the above sets the state of the environment.
        """
        raise NotImplementedError

    def set_state(self, obs):
        """
        Equivalent to the C function above.
        
        set_state_from_ob() should be equivalent to invoking this
        function and then calling self.sim.forward()
        """
        raise NotImplementedError

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip

    def step(self, ctrl):
        self.prestep_callback(np.asarray(ctrl, dtype=float))
        self.sim.step()
        self.poststep_callback()
        obs, reward, done = self.get_obs()
        info = {}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
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
