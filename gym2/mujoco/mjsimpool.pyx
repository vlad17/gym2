ctypedef void (*ObsCopyFn)(double[:], double*, np.uint8_t*, mjModel*, mjData*) nogil
ctypedef void (*PrestepCallback)(mjModel*, mjData*, double[:]) nogil
ctypedef void (*PoststepCallback)(mjModel*, mjData*) nogil

cdef inline void mjstep_with_callbacks(
    mjModel* model, mjData* data, int nsubsteps,
    uintptr_t observation_fn, uintptr_t prestep_callback,
    uintptr_t poststep_callback, double[:] actions,
    double* reward, np.uint8_t* done, double[:] out_obs) nogil:
    cdef int j

    if prestep_callback:
        (<PrestepCallback> prestep_callback)(model, data, actions)
    for j in range(nsubsteps):
        mj_step(model, data)
    if poststep_callback:
        (<PoststepCallback> poststep_callback)(model, data)
    if observation_fn:
        (<ObsCopyFn> observation_fn)(out_obs, reward, done, model, data)

cdef class MjSimPool(object):
    """
    Keeps a pool of multiple MjSims and enables stepping them quickly
    in parallel.

    Parameters
    ----------
    sims : list of :class:`.MjSim`
        List of simulators that make up the pool.
    nsubsteps:
        Number of substeps to run on :meth:`.step`. The individual
        simulators' ``nsubstep`` will be ignored.
    """
    # Arrays of pointers to mjDatas and mjModels for fast multithreaded access
    cdef mjModel** _models
    cdef mjData** _datas
    # Number of frames per step
    cdef int nsubsteps
    # Array of function pointers for mujoco env pre and post step processing
    cdef uintptr_t* _observation_copy_fns # ObsCopyFn
    cdef uintptr_t* _prestep_callbacks # PrestepCallback
    cdef uintptr_t* _poststep_callbacks # PoststepCallback
    # Maximum number of threads to use
    cdef int _max_threads

    """
    The :class:`.MjSim` objects that are part of the pool.
    """
    cdef readonly list sims

    def __cinit__(self, list sims, int nsubsteps=1,
                  list observation_copy_fns=None,
                  list prestep_callbacks=None,
                  list poststep_callbacks=None,
                  int max_threads=1):
        self.sims = sims
        self.nsubsteps = nsubsteps
        self._allocate_data_pointers(
            observation_copy_fns, prestep_callbacks, poststep_callbacks)
        self._max_threads = max_threads

    def reset(self, nsims=None):
        """
        Resets all simulations in pool.
        If :attr:`.nsims` is specified, than only the first :attr:`.nsims` simulators are reset.
        """
        length = self.nsims

        if nsims is not None:
            if nsims > self.nsims:
                raise ValueError("nsims is larger than pool size")
            length = nsims

        for i in range(length):
            self.sims[i].reset()

    def forward(self, nsims=None):
        """
        Calls ``mj_forward`` on all simulations in parallel.
        If :attr:`.nsims` is specified, than only the first :attr:`.nsims` simulator are forwarded.
        """
        cdef int i
        cdef int length = self.nsims

        if nsims is not None:
            if nsims > self.nsims:
                raise ValueError("nsims is larger than pool size")
            length = nsims

        # See explanation in MjSimPool.step() for why we wrap warnings this way
        with wrap_mujoco_warning():
            with nogil, parallel(num_threads=self._num_threads(length)):
                for i in prange(length, schedule='guided'):
                    mj_forward(self._models[i], self._datas[i])

    def step(self,
             np.ndarray actions,
             np.ndarray out_obs,
             np.ndarray out_reward,
             np.ndarray out_done,
             nsims=None,
             np.ndarray mask=None):
        """
        Calls ``mj_step`` on all simulations in parallel, with ``nsubsteps`` as
        specified when the pool was created.

        If :attr:`.nsims` is specified, than only the first :attr:`.nsims` simulator are stepped.

        If the mask is specified, then it is used to only evaluate step() on those environments
        for which the mask is True.

        If the mask is specified, then it is also updated according to the done results.
        """
        cdef int i, j
        cdef int length = self.nsims
        cdef np.uint8_t[:] cmask = mask
        cdef double[:, :] cactions = actions
        cdef double[:, :] cobs = out_obs
        cdef double[:] creward = out_reward
        cdef np.uint8_t[:] cdone = out_done

        if nsims is not None:
            if nsims > self.nsims:
                raise ValueError("nsims is larger than pool size")
            length = nsims

        for sim in self.sims[:length]:
            sim.step_udd()

        # Wrapping these calls to mj_step is tricky, since they're parallelized
        # and can't access the GIL or global python objects.
        # Because we expect to have fatal warnings, we'll just wrap the entire
        # section, and if any call ends up setting an exception we'll raise.
        with wrap_mujoco_warning():
            with nogil, parallel(num_threads=self._num_threads(length)):
                for i in prange(length, schedule='guided'):
                    if i < length and (cmask is None or cmask[i]):
                        mjstep_with_callbacks(
                            self._models[i], self._datas[i], self.nsubsteps,
                            self._observation_copy_fns[i],
                            self._prestep_callbacks[i],
                            self._poststep_callbacks[i],
                            cactions[i], &creward[i], &cdone[i], cobs[i])
                        if cdone[i] and cmask is not None:
                            cmask[i] = 0

    @property
    def nsims(self):
        """
        Number of simulations in the pool.
        """
        return len(self.sims)

    cdef int _num_threads(self, int n) nogil:
        cdef int num_threads = self._max_threads
        if n < self._max_threads:
            num_threads = n
        if num_threads <= 0:
            num_threads = 1
        return num_threads

    @staticmethod
    def create_from_sim(sim, nsims):
        """
        Create an :class:`.MjSimPool` by cloning the provided ``sim`` a total of ``nsims`` times.
        Returns the created :class:`.MjSimPool`.

        Parameters
        ----------
        sim : :class:`.MjSim`
            The prototype to clone.
        nsims : int
            Number of clones to create.
        """
        sims = [MjSim(sim.model, udd_callback=sim.udd_callback)
                for _ in range(nsims)]
        return MjSimPool(sims, nsubsteps=sim.nsubsteps)

    cdef _allocate_data_pointers(self, obs, pre, post):
        self._models = <mjModel**>malloc(self.nsims * sizeof(mjModel *))
        self._datas = <mjData**>malloc(self.nsims * sizeof(mjData *))
        self._observation_copy_fns = <uintptr_t * >malloc(self.nsims * sizeof(uintptr_t))
        self._prestep_callbacks = <uintptr_t * >malloc(self.nsims * sizeof(uintptr_t))
        self._poststep_callbacks = <uintptr_t * >malloc(self.nsims * sizeof(uintptr_t))
        for i in range(self.nsims):
            sim = <MjSim > self.sims[i]
            self._models[i] = sim.model.ptr
            self._datas[i] = sim.data.ptr
            self._observation_copy_fns[i] = obs[i] if obs else 0
            self._prestep_callbacks[i] = pre[i] if pre else 0
            self._poststep_callbacks[i] = post[i] if post else 0

    def __dealloc__(self):
        free(self._datas)
        free(self._models)
        free(self._prestep_callbacks)
        free(self._poststep_callbacks)
        free(self._observation_copy_fns)
