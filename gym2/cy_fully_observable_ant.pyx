"""
This file is included inside the cythonized.pyx module. It contains
the performance-critical sections of the ant environment.

This file should be read as part of fully_observable_ant.py
"""

cdef enum:
    X_COORDS_PER_BODY = 3
    CFRC_COORDS_PER_BODY = 6

# foa = fully observable ant
# userdata[0] - reward
# userdata[1] - done
# userdata[2] - index of ant body in mujoco data structures

# gym code
# def _get_obs(self):
#     return np.concatenate([
#         self.model.data.qpos.flat[1:],
#         self.model.data.qvel.flat,
#     ])

# note the difference from gym, we include both:
# x position
# contact force data
cdef void cython_foa_get_obs(
        double[:] out_obs,
        double* out_reward,
        np.uint8_t* out_done,
        mjModel* model,
        mjData* data) nogil:
    cdef int base = 0
    cdef int i
    cdef int torso_idx = <int>data.userdata[2]
    out_obs[base] = data.xpos[torso_idx * X_COORDS_PER_BODY + 0]
    base += 1
    for i in range(model.nq):
        out_obs[base + i] = data.qpos[i]
    base += model.nq
    for i in range(model.nv):
        out_obs[base + i] = data.qvel[i]
    base += model.nv
    for i in range(CFRC_COORDS_PER_BODY * model.nbody):
        out_obs[base + i] = data.cfrc_ext[i]
    out_reward[0] = data.userdata[0]
    out_done[0] = data.userdata[1] > 0

def python_foa_c_get_obs_fn():
    return <uintptr_t>&cython_foa_get_obs

def python_foa_obs_shape(model):
    cdef int cfrc_size = CFRC_COORDS_PER_BODY * model.nbody
    return (1 + model.nv + model.nq + cfrc_size,)

def python_foa_get_obs(sim):
    obs = np.empty(python_foa_obs_shape(sim.model))
    cdef double[:] cobs = obs
    cdef double rew
    cdef np.uint8_t done
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_foa_get_obs(cobs, &rew, &done, model.ptr, data.ptr)
    return obs, rew, done

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

cdef void cython_foa_prestep(
    mjModel* model, mjData* data, double[:] actions) nogil:
    data.userdata[0] = data.qpos[0]
    cdef int i
    for i in range(model.nu):
        data.ctrl[i] = actions[i]

def python_foa_c_prestep_fn():
    return <uintptr_t>&cython_foa_prestep

def python_foa_prestep(sim, actions):
    cdef double[:] cactions = actions
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_foa_prestep(model.ptr, data.ptr, cactions)

cdef int FOA_FRAMESKIP = 5

from libc.math cimport isfinite

cdef void cython_foa_poststep(mjModel* model, mjData* data) nogil:
    cdef int cfrc_size = CFRC_COORDS_PER_BODY * model.nbody
    cdef double x_before = data.userdata[0]
    cdef double x_after = data.qpos[0]

    cdef double truncated_cfrc_magnitude = 0
    cdef int i
    for i in range(cfrc_size):
      truncated_cfrc_magnitude += data.cfrc_ext[i] * data.cfrc_ext[i]

    cdef double control_magnitude = 0
    cdef double dt = model.opt.timestep * FOA_FRAMESKIP
    for i in range(model.nu):
      control_magnitude += data.ctrl[i] * data.ctrl[i]

    data.userdata[0] = (x_after - x_before) / dt
    data.userdata[0] -= 0.5 * control_magnitude
    data.userdata[0] -= 0.5 * 1.0e-3 * truncated_cfrc_magnitude
    data.userdata[0] += 1.0

    data.userdata[1] = 0
    for i in range(model.nq):
        if not isfinite(data.qpos[i]):
            data.userdata[1] = 1

    for i in range(model.nv):
        if not isfinite(data.qvel[i]):
            data.userdata[1] = 1

    if data.qpos[2] < 0.2 or data.qpos[2] > 1:
        data.userdata[1] = 1


def python_foa_poststep(sim):
    cdef PyMjData data = sim.data
    cdef PyMjModel model = sim.model
    cython_foa_poststep(model.ptr, data.ptr)

def python_foa_c_poststep_fn():
    return <uintptr_t>&cython_foa_poststep

def python_foa_frameskip():
    return FOA_FRAMESKIP
