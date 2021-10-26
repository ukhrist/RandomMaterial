# distutils: language = c++


cimport cython

cimport numpy as np
import numpy as np
from time import time 

from cython.parallel import prange, parallel



#########################################################################################
#                         Linear elasticity Green's operator
#########################################################################################

cdef extern from "cpplib.cpp":
    cdef void construct_GreenHat(double* G, int* Nd0, int* Nd, int d, double lmbda, double mu, double* frq, int Ntrunc)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def construct_FourierOfPeriodizedGreenOperator( double lmbda, double mu,
                                                np.ndarray[int, ndim=1] Nd0, np.ndarray[int, ndim=1] Nd, int d,
                                                np.ndarray[double, ndim=1] frq, int Ntrunc):
    cdef int nVoigt = int(d*(d+1)/2)
    cdef np.ndarray[double, ndim=1, mode="c"] G_hat = np.zeros(np.prod(Nd-Nd0) * nVoigt*nVoigt)
    construct_GreenHat(&G_hat[0], &Nd0[0], &Nd[0], d, lmbda, mu, &frq[0], Ntrunc)
    G_hat /= mu # 1/mu from Green Operator
    return G_hat

#########################################################################################

cdef extern from "cpplib.cpp":
    cdef void apply_GreenOperator(double* G, int* Nd, int d, complex* tau_hat, complex* eta_hat)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
# def apply_Green_cpp(np.ndarray[double, ndim=1, mode="c"] GreenOp,
#                     np.ndarray[complex, ndim=1, mode="c"] tau_hat,
#                     np.ndarray[int, ndim=1] Nd, int n,
#                     np.ndarray[complex, ndim=1, mode="c"] eta_hat):
def apply_Green_cpp(np.ndarray[double, ndim=1, mode="c"] GreenOp,
                    # np.ndarray[complex, ndim=1, mode="c"] tau_hat,
                    tau_hat,
                    np.ndarray[int, ndim=1] Nd, int n,
                    np.ndarray[complex, ndim=1, mode="c"] eta_hat):
    cdef long p_tau_hat = tau_hat.__array_interface__['data'][0]
    apply_GreenOperator(&GreenOp[0], &Nd[0], n, <complex*>p_tau_hat, &eta_hat[0])
    # apply_GreenOperator(&GreenOp[0], &Nd[0], n, &tau_hat[0], &eta_hat[0])

#########################################################################################

cdef extern from "cpplib.cpp":
    cdef void apply_Matrix(int* Phase, double* C1, double* C2, int* Nd, int n, double* x, double* y)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def apply_Matrix_cpp(   Phase,
# np.ndarray[int, ndim=1, mode="c"] Phase,
                            np.ndarray[double, ndim=1, mode="c"] C1,
                            np.ndarray[double, ndim=1, mode="c"] C2,
                            np.ndarray[int, ndim=1] Nd, int n,
                            np.ndarray[double, ndim=1, mode="c"] x,
                            np.ndarray[double, ndim=1, mode="c"] y):
    cdef long p_Phase = Phase.__array_interface__['data'][0]
    apply_Matrix(<int*>p_Phase, &C1[0], &C2[0], &Nd[0], n, &x[0], &y[0])
    # apply_Matrix(&Phase[0], &C1[0], &C2[0], &Nd[0], n, &x[0], &y[0])

#########################################################################################




#########################################################################################
#                            Largest inclusion search
#########################################################################################

cdef extern from "cpplib.cpp":
    cdef int findLargestInclusionVicinity(int* Phase, int* Nd, int d, int* indexIncl, int* markerLargestIncl, int vicinity_depth, bint vicinity_only)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def getLargestInclusionVicinity(np.ndarray[int, ndim=1, mode="c"] Phase,
                                np.ndarray[int, ndim=1] Nd, int d,
                                np.ndarray[int, ndim=1, mode="c"] indexIncl,
                                np.ndarray[int, ndim=1, mode="c"] markerLargestIncl,
                                double vicinity_depth = 0, bint vicinity_only = False):    
    return findLargestInclusionVicinity(&Phase[0], &Nd[0], d, &indexIncl[0], &markerLargestIncl[0], int(vicinity_depth*Nd[0]), vicinity_only)


#########################################################################################
#                            All Inclusion search (volume, surface)
#########################################################################################

cdef extern from "cpplib.cpp":
    cdef int findAllInclusionsVolumesAndSurfaces(int* Phase, int* Nd, int d, int* indexIncl, int* volume, double* surface)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def getAllInclusionsVolumesAndSurfaces(Phase):
    cdef int d = Phase.ndim
    cdef np.ndarray[int, ndim=1, mode="c"] Nd = np.array(Phase.shape, dtype=np.intc)
    cdef np.ndarray[int, ndim=1, mode="c"] indexIncl = np.zeros(Phase.size, dtype=np.intc)
    cdef np.ndarray[int, ndim=1, mode="c"] volume    = np.zeros(Phase.size, dtype=np.intc)
    cdef np.ndarray[double, ndim=1, mode="c"] surface = np.zeros(Phase.size, dtype=np.float)
    cdef np.ndarray[int, ndim=1, mode="c"] cPhase = Phase.astype(np.intc).flatten()
    nIncl = findAllInclusionsVolumesAndSurfaces(&cPhase[0], &Nd[0], d, &indexIncl[0], &volume[0], &surface[0])
    return indexIncl.reshape(Phase.shape), volume[:nIncl], surface[:nIncl]