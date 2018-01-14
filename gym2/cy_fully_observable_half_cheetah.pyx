"""
This file is included inside the cythonized.pyx module. It contains
the performance-critical sections of the half cheetah environment.

This file should be read as part of fully_observable_half_cheetah.py
"""

# fohc = fully observable half cheetah
# userdata[0] - reward
# userdata[1] - done

# gym code
# def _get_obs(self):
#     return np.concatenate([
#         self.model.data.qpos.flat[1:],
#         self.model.data.qvel.flat,
#     ])

# note the difference from gym, we include the x position
cdef void cython_fohc_get_obs(
        double[:] out_obs,
        double* out_reward,
        np.uint8_t* out_done,
        mjModel* model,
        mjData* data) nogil:
    cdef int i
    for i in range(model.nq):
        out_obs[i] = data.qpos[i]
    for i in range(model.nv):
        out_obs[i + model.nq] = data.qvel[i]
    out_reward[0] = data.userdata[0]
    out_done[0] = data.userdata[1] > 0

def python_fohc_c_get_obs_fn():
    return <uintptr_t>&cython_fohc_get_obs

def python_fohc_obs_shape(model):
    return (model.nv + model.nq,)

def python_fohc_get_obs(sim):
    obs = np.empty(python_fohc_obs_shape(sim.model))
    cdef double[:] cobs = obs
    cdef double rew
    cdef np.uint8_t done
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_fohc_get_obs(cobs, &rew, &done, model.ptr, data.ptr)
    return obs, rew, done

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

cdef void cython_fohc_prestep(
    mjModel* model, mjData* data, double[:] actions) nogil:
    data.userdata[0] = data.qpos[0]
    cdef int i
    for i in range(model.nu):
        data.ctrl[i] = actions[i]

def python_fohc_c_prestep_fn():
    return <uintptr_t>&cython_fohc_prestep

def python_fohc_prestep(sim, actions):
    cdef double[:] cactions = actions
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_fohc_prestep(model.ptr, data.ptr, cactions)

cdef int FOHC_FRAMESKIP = 5

cdef void cython_fohc_poststep(mjModel* model, mjData* data) nogil:
    cdef double x_before = data.userdata[0]
    cdef double x_after = data.qpos[0]
    cdef double control_magnitude = 0
    cdef double dt = model.opt.timestep * FOHC_FRAMESKIP
    cdef int i
    for i in range(model.nu):
        control_magnitude += data.ctrl[i] * data.ctrl[i]
    data.userdata[0] = (x_after - x_before) / dt - 0.1 * control_magnitude
    data.userdata[1] = 0

def python_fohc_poststep(sim):
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_fohc_poststep(model.ptr, data.ptr)

def python_fohc_c_poststep_fn():
    return <uintptr_t>&cython_fohc_poststep

def python_fohc_frameskip():
    return FOHC_FRAMESKIP
