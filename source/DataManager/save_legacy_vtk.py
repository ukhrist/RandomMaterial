
import numpy as np
import torch

def save_legacy_vtk(filename, data):
    if torch.is_tensor(data):
        data = data.detach().numpy()
    
    header = """# vtk DataFile Version 4.2
vtk output
ASCII
DATASET STRUCTURED_POINTS
"""

    shape       = data.shape
    ndim        = len(shape)
    shape_points= list(np.array(shape)+1)
    shape_cells = list(np.array(shape)-1)
    spacing     = (1./min(shape_cells), ) * ndim
    origin      = (0, ) * ndim

    content = header
    content += "DIMENSIONS {0:d} {1:d} {2:d}\n".format(*shape)
    content += "SPACING {0:f} {1:f} {2:f}\n".format(*spacing)
    content += "ORIGIN {0:d} {1:d} {2:d}\n".format(*origin)
    content += "POINT_DATA {0:d}\n".format(data.size)
    content += "SCALARS field double\n"
    content += "LOOKUP_TABLE default\n"

    counter = 0
    for x in data.ravel(order='F'):
        content += "{0:d} ".format(int(x))
        counter += 1
        if counter==9:
            content += "\n"
            counter = 0

    content += """METADATA
INFORMATION 0\n
"""

    with open(filename+".vtk", 'w+') as f:
        f.write(content)