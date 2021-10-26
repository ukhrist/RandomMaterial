
import torch

from source.RMNet.DataManager import read_data3D
from source.RMNet.MultiPhaseFields.MultiPhaseMaterial import MultiGrainMaterial
from source.RMNet.Kernels.MaternKernel import MaternKernel



config = {
    'ndim'              :   2,
    'grid_level'        :   8,
    'window_margin'     :   0,
    'vf'                :   0.5,
### Covariance
    'GRF_covariance'    :   MaternKernel,
    'nu'                :   2,
    'correlation_length':   0.02,
    'Folded_GRF'        :   False,
### Hibrid
    'alpha'             :   0*0.05,
    'tau'               :   0,
### Grains
    'nPhases'           :   10,
    'SizeCell'          :   [80,40],
    'mask'              :   True,
### Data
    'input_folder'  :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/grains/Images_EBSD_Solenne/',
    'output_folder' :   '/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/grains/output/', 
    'export_np'     :   True,
    'export_vtk'    :   True,  
    'crop'          :   [0.15, 0.8, 0.15, 0.8, 0.3, 0.7],
    'threshold'     :   0.5,
}

# read_data3D(**config)

RM = MultiGrainMaterial(**config)
RM.save_vtk(config['output_folder'] + 'sample')

RM.plot()

