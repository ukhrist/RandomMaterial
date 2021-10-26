
import numpy as np
from LinearElasticity import cpplib

def getLargestInclusionVicinity(Phase, vicinity_depth, vicinity_only=False):

    ndim = Phase.ndim
    Nd = np.array(Phase.shape, dtype=np.intc)
    nPoints = np.prod(Nd)

    indexIncl = np.zeros(nPoints, dtype=np.intc)
    subdomain = np.zeros(nPoints, dtype=np.intc)

    cpplib.getLargestInclusionVicinity(Phase.astype(np.intc).flatten(), Nd, ndim, indexIncl, subdomain, vicinity_depth)

    indexIncl = indexIncl.reshape(Phase.shape)
    subdomain = subdomain.reshape(Phase.shape)

    if vicinity_only: subdomain -= subdomain*Phase

    return subdomain, indexIncl