cdef void callback_step(mjModel * model, mjData * data, int nsubsteps,
                        uintptr_t prestep_callback,
                        uintptr_t poststep_callback) nogil:
    if prestep_callback:
        ( < mjfGeneric > prestep_callback)(model, data)
    for _ in range(nsubsteps):
        mj_step(model, data)
    if poststep_callback:
        ( < mjfGeneric > poststep_callback)(model, data)

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
    cdef mjModel ** _models
    cdef mjData ** _datas
    # Array of function pointers for pre and post step processing
    cdef uintptr_t * _prestep_callbacks
    cdef uintptr_t * _poststep_callbacks
    # Number of frames per step
    cdef int nsubsteps

    """
    The :class:`.MjSim` objects that are part of the pool.
    """
    cdef readonly list sims

    def __cinit__(self, list sims, int nsubsteps=1):
        self.sims = sims
        self.nsubsteps = nsubsteps
        self._allocate_data_pointers()

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
            with nogil, parallel():
                for i in prange(length, schedule='guided'):
                    mj_forward(self._models[i], self._datas[i])

    def step(self, nsims=None, np.ndarray mask=None):
        """
        Calls ``mj_step`` on all simulations in parallel, with ``nsubsteps`` as
        specified when the pool was created.

        If :attr:`.nsims` is specified, than only the first :attr:`.nsims` simulator are stepped.

        If the mask is specified, then it is used to only evaluate step() on those environments
        for which the mask is True
        """
        cdef int i, j
        cdef int length = self.nsims
        cdef np.ndarray[np.uint8_t, cast = True] cmask = mask

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
            with nogil, parallel():
                for i in prange(length, schedule='guided'):
                    # TODO if cmask
                    callback_step(self._models[i], self._datas[i], self.nsubsteps,
                                  self._prestep_callbacks[i], self._poststep_callbacks[i])

    @property
    def nsims(self):
        """
        Number of simulations in the pool.
        """
        return len(self.sims)

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

    cdef _allocate_data_pointers(self):
        self._models = <mjModel**>malloc(self.nsims * sizeof(mjModel *))
        self._datas = <mjData**>malloc(self.nsims * sizeof(mjData *))
        self._prestep_callbacks = <uintptr_t * >malloc(self.nsims * sizeof(uintptr_t))
        self._poststep_callbacks = <uintptr_t * >malloc(self.nsims * sizeof(uintptr_t))
        for i in range(self.nsims):
            sim = <MjSim > self.sims[i]
            self._models[i] = sim.model.ptr
            self._datas[i] = sim.data.ptr
            self._prestep_callbacks[i] = sim.prestep_callback_ptr
            self._poststep_callbacks[i] = sim.poststep_callback_ptr

    def __dealloc__(self):
        free(self._datas)
        free(self._models)
        free(self._prestep_callbacks)
        free(self._poststep_callbacks)
