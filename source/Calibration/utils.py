

import numpy as np


    
def save_as_csv(data, fileneame):
    a = np.asarray(data)
    if a.ndim==1: a = a.reshape([-1, 1])
    np.savetxt(fileneame+".csv", a, delimiter=",")