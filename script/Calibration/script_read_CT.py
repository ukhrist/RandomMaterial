
from math import pi
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.DataManager import read_data3D, save_legacy_vtk

i, j, k = 1, 1, 1
L, Lz = 0.315, 0.225
a, b = 0.32 - L/2 + i*L, 0.635 - L/2 + i*L
c, d = 0.32 - L/2 + j*L, 0.635 - L/2 + j*L
e, f = 0.365 + k*Lz, 0.585 + k*Lz

a, b, c, d, e, f = 0.2, 0.8, 0.2, 0.8, 0.2, 0.8

config = {
    'input_folder'      :   "/home/ustim/Projects/Paris/data_LMS/lattice/SlicesY_8bit/", #'/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/200N/SlicesY_8bit/',
    'ext'               :   'tif',
    'output_folder'     :   "/home/ustim/Projects/Paris/data_LMS/lattice/workfolder/", #'/home/khristen/Projects/Paris/RandomMaterialCode/data/case_LMS/lattice/workfolder/',
    'output_filename'   :   'data3D_full', #'data3D_cell12',
    'export_np'         :   False,
    'export_vtk'        :   False,
    # 'crop'              :   [a, b] + [c,d] + [e,f], ### ratio
    'threshold'         :   0.4, #0.4, ### ratio
}


data3D = read_data3D(**config)

# new_shape = [100, 100, 150] #np.array(data3D.shape) // 5
# data3D_tensor = torch.tensor(data3D, dtype=torch.double).unsqueeze(0).unsqueeze(0)
# Y = torch.nn.functional.interpolate(data3D_tensor, size=tuple(new_shape)).squeeze()

x = np.linspace(0, 1, data3D.shape[0])
y = np.linspace(0, 1, data3D.shape[1])
z = np.linspace(0, 1, data3D.shape[2])
my_interpolating_function = RegularGridInterpolator((x, y, z), data3D, method="nearest")

new_shape = [300, 300, 450]
x = np.linspace(0, 1, new_shape[0])
y = np.linspace(0, 1, new_shape[1])
z = np.linspace(0, 1, new_shape[2])
x, y, z = np.meshgrid(x, y, z, indexing="ij")

pts = np.stack([x.ravel(), y.ravel(), z.ravel()]).T
data3D = my_interpolating_function(pts).reshape(new_shape)


filename = config['output_folder'] + config['output_filename']
save_legacy_vtk(filename, data3D)
