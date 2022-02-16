
from math import pi, sqrt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.StructureFields.SupportedMaterials import LatticeMaterial
from source.RandomParameters import RandomParameters
from source.DataManager.save_legacy_vtk import save_legacy_vtk


###=============================================
### Configuration
from config import config

fg_verbose           = config['verbose']
fg_random_parameters = config['fg_random_parameters']
fg_fixed_seed        = config['fg_fixed_seed']

EXPORTDIR = "./"
filename  = os.path.abspath(os.path.join(EXPORTDIR, "sample_OctetLatticeCell"))

###=============================================

if fg_verbose:
    print()
    print("=======================================")
    print(" Sampling")
    print("=======================================")


### Parameters randomizer
RM = LatticeMaterial(**config)
RP = RandomParameters(config['model_parameters'])

### fix random seed (for testing)
if fg_fixed_seed: RM.seed(0)

### Generate a sample
if fg_random_parameters:
    RP.sample_parameters(RM)
    if fg_verbose: print("Sampled parameters..")
Sample = RM.sample_numpy()

### Info
print()
print('vf = ', Sample.mean())
print('shape: ', Sample.shape)
print()

###=============================================
### Export as vtk
RM.save_vtk(filename, Sample)
print(f"Successfully saved as {filename}.vti")


#######################################################

plt.show()




