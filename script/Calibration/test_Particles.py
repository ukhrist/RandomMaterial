
import torch
from source.RMNet.DataManager import read_data3D
# from source.RMNet.StructureFields.Lattice2 import LatticeStructure
from source.RMNet.StructureFields.SupportedMaterials import LatticeMaterial, GrainMaterial, ParticlesMaterial
from source.RMNet.Kernels.MaternKernel import MaternKernel

import matplotlib.pyplot as plt


config = {
    'ndim'              :   2,
    'grid_level'        :   8,
    'window_margin'     :   0,
    'vf'                :   0.5,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   1,
    'correlation_length':   0.02,
    'Folded_GRF'        :   True,
### Hibrid
    'alpha'             :   0.2,
    'tau'               :   0,
### Particles
    'radius'            :   0.03,
    'Poisson_mean'      :   30,
### Data
    'input_folder'  :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_ParticlesSize/samples_original/',  ### DO NOT FORGET "/" in the end of the path !!!
    'output_folder' :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_ParticlesSize/output/', 
    'export_np'     :   True,
    'export_vtk'    :   True,  
    'crop'          :   [0.16, 0.48, 0.16, 0.48, 0.35, 0.6],
    'threshold'     :   0.5,
}

# read_data3D(**config)

RM = ParticlesMaterial(**config)
RM.save_vtk(config['output_folder'] + 'sample')

if RM.ndim==2:
    plt.imshow(RM.sample())
    plt.show()
