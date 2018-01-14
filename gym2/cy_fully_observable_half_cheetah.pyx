"""
This file is included inside the cythonized.pyx module. It contains
the performance-critical sections of the half cheetah environment.
"""

# TODO bounds checks, misc

# return value is whether done


cpdef bint fully_observable_half_cheetah_get_obs(
        double[:] out_obs,
        double[:] out_reward,
        np.uint8_t[:] out_done,
        uintptr_t model_ptr, uintptr_t data_ptr) nogil:
    cdef mjModel* model = <mjModel*>model_ptr
    cdef mjData* data = <mjData*>data_ptr
    cdef int i
    for i in range(model.nu):
        out_obs[i] = data.qpos[i]
    for i in range(model.nv):
        out_obs[i + model.nu] = data.qvel[i]
    out_reward[0] = data.userdata[0]
    out_done[0] = data.userdata[1] > 0
