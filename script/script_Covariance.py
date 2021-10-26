
import numpy as np
from math import *
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

import sys
# sys.path.append("/home/khristen/Projects/Paris/random_material/code/source/")
sys.path.append("/home/khristen/Projects/Paris/RandomMaterialCode/source/")

from SurrogateMaterialModeling.utilities.DataLoader import DataLoader
from SurrogateMaterialModeling.utilities.statistics import compute_covariance

#######################################################

fg_load = True
if fg_load:
    DL = DataLoader(folder='/home/khristen/Projects/Paris/RandomMaterialCode/samples/covariance')
else:
    DL = DataLoader(folder='/home/khristen/Projects/Paris/RandomMaterialCode/samples')

i = 0
for X in DL():
    if fg_load:
        Cov = X
    else:
        Cov = compute_covariance(X)
        np.save(DL.Folder+'covariance/covariance_{0:d}'.format(i), Cov)
    plt.plot(Cov/Cov[0], label='Sample {0:d}: vf={1:1.3f}'.format(i, Cov[0]))
    i = i+1

#######################################################

# plt.xscale('log')
plt.yscale('log')
plt.xlabel('r')
plt.ylabel('Correlation')
plt.legend()
plt.show()