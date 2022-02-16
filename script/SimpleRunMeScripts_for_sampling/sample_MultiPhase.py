from math import pi, sqrt
from turtle import color
import matplotlib
from tqdm import tqdm
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.DataManager import read_data3D
from source.MultiPhaseFields.MultiPhaseMaterial import MultiGrainMaterial
from source.Kernels.MaternKernel import MaternKernel


###=============================================
### Configuration parameters

config = {
    'ndim'              :   2,
    'grid_level'        :   9,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   1,
    'correlation_length':   0.02,
    'Folded_GRF'        :   False,
### Support
    'alpha'             :   0.8,
### Grains
    'nPhases'           :   10,
    'SizeCell'          :   [40,128],
    'angle'             :   pi/3,
    'mask'              :   False,
}

### Sampling
nsamples = 20
EXPORTDIR = "./"


###=============================================
### Main part

RM = MultiGrainMaterial(**config)
# RM.seed(0) ### fix the random seed in order to have always the same realizations 

### Generate multiple samples
for isample in tqdm(range(nsamples)):
    filename = os.path.abspath(os.path.join(EXPORTDIR, f"sample_{isample+1}.png"))
    RM.save_png(filename)


### Generate one nice looking sample
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['image.cmap']='rainbow'
Sample = RM.sample_numpy()
plt.imshow(Sample)
plt.contour(Sample, config['nPhases'], colors="black", linewidths=0.3)
plt.axis('off')
filename  = os.path.abspath(os.path.join(EXPORTDIR, "sample_MF.jpg"))
plt.savefig(filename, bbox_inches='tight')



print(f"Samples are successfully saved in {os.path.abspath(EXPORTDIR)}")



###=============================================
plt.show()

