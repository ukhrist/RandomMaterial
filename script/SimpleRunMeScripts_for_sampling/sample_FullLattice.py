
from math import pi, sqrt
import numpy as np
from tqdm import tqdm
import importlib
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
from source.Kernels.MaternKernel import MaternKernel
from source.RandomParameters import RandomParameters
from source.DataManager.save_legacy_vtk import save_legacy_vtk

### Configuration

EXPORTDIR = "./"
filename  = os.path.abspath(os.path.join(EXPORTDIR, "sample_OctetLatticeFull"))

ncells = 8

config = {
#################################################
#                   General
#################################################

    'dev'               :   True,
    'verbose'           :   True,
    'debug'             :   True,
    'export_vtk'        :   False,

    'fg_fixed_seed'     :   False,
    'vtk_format'        :   "default", ### "legacy", "default"

#################################################
#            Surrogate Material
#################################################

    'ndim'              :   3,
    'grid_level'        :   9,
    'window_margin'     :   0,
    'vf'                :   0.2,

    ### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   2,
    'correlation_length':   0.04/ncells,
    'Folded_GRF'        :   False,

    ### Lattice
    'nCells'            :   ncells,
    'SizeVoid'          :   0,
    'alpha'             :   0.015/ncells,
    'thickness'         :   0.15/ncells,
    'nodal_pores'       :   False,

}


if config['verbose']:
    print()
    print("===============================================")
    print(" Sampling a full block of octet-truss lattice")
    print("===============================================")


RM = LatticeMaterial(**config)
Sample = RM.sample_numpy()

### Output the volume fraction of the sample
vf = Sample.mean()
print(f"Volume fraction: {vf}")

### Reshape (optional)
n, m = Sample.shape[0]//2, Sample.shape[0]//4
X = Sample[:, :n, :m]
Sample = np.concatenate([X]*2)

if config['vtk_format'] == "legacy":
    save_legacy_vtk(filename, Sample)
else:
    RM.save_vtk(filename, Sample)
    
if config['verbose']:
    print(f"Sample is saved as {filename}")
    print("=======================================")
    print()




