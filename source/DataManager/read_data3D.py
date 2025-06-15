
import sys, os, glob
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from .utils import exportVTK

"""
==================================================================================================================
Data Reader
==================================================================================================================
"""

def read_data3D(**kwargs):
    verbose         = kwargs.get('verbose', False)
    input_folder    = kwargs.get('input_folder', './')
    output_folder   = kwargs.get('output_folder', input_folder)
    output_filename = output_folder + kwargs.get('output_filename', 'data3D')
    ext             = kwargs.get('ext', 'tif')
    export_np       = kwargs.get('export_np',  False)
    export_vtk      = kwargs.get('export_vtk', False)
    crop            = kwargs.get('crop', [0,1,0,1,0,1]) ### ratio
    threshold       = kwargs.get('threshold', None) ### ratio


    filenames = glob.glob(input_folder + "*." + ext)
    filenames = sorted(filenames, key=key_func)
    nlayers   = len(filenames)

    if type(crop)==list:
        fg_crop = True
        margins = crop[:4]   ### margins of a 2D layer
        begin, end = int(nlayers*crop[4]), int(nlayers*crop[5])
    else:
        fg_crop = False
        begin, end = 0, nlayers

    print("\n\nReading data from {0:s}".format(input_folder))

    for ilayer, filename in enumerate(tqdm(filenames[begin:end], total=nlayers)):

        ### Read the layer
        im = Image.open(filename) #.convert('1')
        imarray = np.array(im) #, dtype=np.bool)
        # plt.imshow(imarray)
        # im.show()

        ### Shape of the layer
        if 'shape_layer' not in locals():
            shape_layer = imarray.shape
        else:
            assert(imarray.shape[0]==shape_layer[0] and imarray.shape[1]==shape_layer[1])

        ### Crop the layer
        if fg_crop:
            n1, n2 = int(shape_layer[0]*margins[0]), int(shape_layer[0]*margins[1])
            m1, m2 = int(shape_layer[1]*margins[2]), int(shape_layer[1]*margins[3])
            # print(n2-n1, m2-m1)
            imarray = imarray[n1:n2, m1:m2]

        ### Concatenate the layer with the 3D array
        if 'data3D' not in locals():
            data3D = np.reshape(imarray, imarray.shape + (1,))
        else:
            data3D = np.concatenate((data3D, imarray[:,:,None]), axis=-1)

    ### Thresholding
    if threshold:
        threshold_value = data3D.min() + threshold * (data3D.max()-data3D.min())
        data3D = 1*(data3D >= threshold_value)

    ### Exports
    if export_np:  np.save(output_filename, data3D)
    if export_vtk: exportVTK(output_filename, {"data3D" : 1*data3D})

    return data3D


"""
==================================================================================================================
Filter key for the filenames
==================================================================================================================
"""

def key_func(fullpath):
    path, filename_with_ext = os.path.split(fullpath)
    name, ext = os.path.splitext(filename_with_ext)
    return int(re.findall(r'\d+', name)[0])
    # if ext in ('tif',):        
    # else:
    #     return None

# def filter_func(fullpath):
#     _, ext = os.path.splitext(fullpath)
# 	return ( ext in ('tif',) )




