
import numpy as np
import pickle



datafile     = "data_stiffness_RP_grid7"
outputfolder = "/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/UQ_MacroStiffness/"


"""
==================================================================================================================
Load data from file
==================================================================================================================
"""

filename = outputfolder + datafile + ".pkl"
with open(filename, 'rb') as filehandler:
    # data, *_ = pickle.load(filehandler)
    data = pickle.load(filehandler)



"""
==================================================================================================================
Comupte Mean and Variance
==================================================================================================================
"""

data = np.array(data, dtype=np.float)
avg  = data.mean(axis=0)
std  = data.std(axis=0)

print("datafile : ", datafile)
print("Mean = ", avg)
print("Std  = ", std)