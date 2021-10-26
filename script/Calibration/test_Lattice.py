
import torch
from source.RMNet.DataManager import read_data3D
from source.RMNet.StructureFields.Lattice2 import LatticeStructure
from source.RMNet.StructureFields.SupportedMaterials import LatticeMaterial, GrainMaterial, ParticlesMaterial
from source.RMNet.Kernels.MaternKernel import MaternKernel



config = {
    'ndim'              :   3,
    'grid_level'        :   8,
    'window_margin'     :   0,
    'vf'                :   0.5,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   4,
    'correlation_length':   0.01,
    'Folded_GRF'        :   False,
### Hibrid
    'alpha'             :   0.01,
    'tau'               :   0,
### Lattice
    'nCells'            :   4,
    'SizeVoid'          :   0,
    'thickness'         :   0.02,
### Data
    'input_folder'  :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/200N/SlicesY_8bit/',
    'output_folder' :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/output/', 
    'export_np'     :   True,
    'export_vtk'    :   True,  
    'crop'          :   [0.16, 0.48, 0.16, 0.48, 0.35, 0.6], ### One cell
    # 'crop'          :   [0.16, 0.48, 0.16, 0.48, 0.35, 0.6],
    'threshold'     :   0.5,
}

# read_data3D(**config)

RM = LatticeMaterial(**config)
RM.save_vtk(config['output_folder'] + 'sample')

