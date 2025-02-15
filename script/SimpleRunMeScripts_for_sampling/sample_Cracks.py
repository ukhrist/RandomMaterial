
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

from source.StructureFields.SupportedMaterials import CracksMaterial
from source.Kernels.MaternKernel import MaternKernel


###=============================================
### Configuration parameters

config = {
    'ndim'              :   2,
    'grid_level'        :   9,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   0.5,
    'correlation_length':   0.04,
    'Folded_GRF'        :   False,
### Support
    'alpha'             :   0.001,
    'thickness'         :   0.001,
    'Poisson_mean'      :   10,
}

fg_verbose           = True
fg_fixed_seed        = True

### Sampling
nsamples = 1

EXPORTDIR = os.path.join(os.path.dirname(__file__), "samples/cracks_labeled") ###NOTE: set your own export folder

opt1 = "a=" + str(config['alpha']).replace('.', '_')
opt2 = "l=" + str(config['correlation_length']).replace('.', '_')
case = opt1 + "__" + opt2
EXPORTDIR = os.path.join(EXPORTDIR, case)
if not os.path.exists(EXPORTDIR): os.makedirs(EXPORTDIR)

settingsfilename = os.path.abspath(os.path.join(EXPORTDIR, "settings.txt"))
with open(settingsfilename, 'w') as file:
    N = int(2**config['grid_level'])
    ngrains, th, a, l, nu = config['Poisson_mean'], config['thickness'], config['alpha'], config['correlation_length'], config['nu']
    file.write(f"Resolution : {N}x{N}x{N}\n")
    file.write(f"Covariance : Matèrn\n")
    file.write(f"Folded : No\n")
    file.write(f"Average number of grains : {ngrains}\n")
    file.write(f"Thickness : {th}\n")
    file.write(f"Perturbation level (alpha) : {a}\n")
    file.write(f"Correlation length : {l}\n")
    file.write(f"Surface regularity (nu) : {nu}\n")

basename = os.path.abspath(os.path.join(EXPORTDIR, "cracks")) 

###=============================================

if fg_verbose:
    print()
    print("=======================================")
    print(" Sampling")
    print("=======================================")

RM = CracksMaterial(**config)

### fix random seed (for testing)
if fg_fixed_seed: RM.seed(0)

### Generate samples
for isample in tqdm(range(nsamples)):
    filename = basename + f"_sample_{isample+1}"
    Sample   = RM.sample_numpy()
    RM.save_numpy(filename, Sample)
    RM.save_vtk(filename, Sample)
    if config['ndim']==2: RM.save_png(filename+".png", Sample)

if fg_verbose:
    print(f"Samples are successfully saved in {EXPORTDIR}")


#######################################################





