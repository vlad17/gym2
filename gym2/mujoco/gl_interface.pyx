cdef extern from "gl/glshim.h":

    cdef int usingEGL()
    cdef int initOpenGL(int device_id)
    cdef void closeOpenGL()
    cdef int makeOpenGLContextCurrent(int device_id)
    cdef int setOpenGLBufferSize(int device_id, int width, int height)

    cdef unsigned int createPBO(int width, int height, int batchSize, int use_short)
    cdef void freePBO(unsigned int pixelBuffer)
    cdef void copyFBOToPBO(mjrContext * con,
                           unsigned int pbo_rgb, unsigned int pbo_depth,
                           mjrRect viewport, int bufferOffset)
    cdef void readPBO(unsigned char * buffer_rgb, unsigned short * buffer_depth,
                      unsigned int pbo_rgb, unsigned int pbo_depth,
                      int width, int height, int batchSize)
