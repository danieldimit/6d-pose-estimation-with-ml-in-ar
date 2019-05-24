import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
warnings.filterwarnings("ignore")

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

def valid(ply):
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(ply)
    diam          = calc_pts_diameter(np.array(mesh.vertices))
    print(diam)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        ply = sys.argv[1]
        valid(ply)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
