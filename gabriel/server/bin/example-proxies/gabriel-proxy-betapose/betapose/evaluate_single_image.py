import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from .opt import opt

from .dataloader import ImageLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from .yolo.util import write_results, dynamic_write_results
from .KPD.src.main_fast_inference import *

import yaml
import os
import sys
import pickle
from tqdm import tqdm
import time
from .fn import getTime
from .utils.model import * # 3D model class
from .utils.sixd import load_sixd
from .utils.metrics import *
from .utils.boundingbox.bb_projector import get_ply_bb_corners, project_obj_onto_img
from queue import Queue, LifoQueue

from .pPose_nms import pose_nms, write_json
from IPython import embed
args = opt
args.dataset = 'coco'
TOTAL_KP_NUMBER = args.nClasses
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings("ignore")

writer = None
det_processor = None
det_loader = None
pose_model = None
ply_model_corners = None
cam_K = None
image_loader = Queue(maxsize=1)




def evaluate_img(img, frame):
    global image_loader, writer, det_processor, det_loader, pose_model, ply_model_corners, cam_K

    # Put a new image on the queue
    image_loader.putitem(img, frame)

    # start inference
    start_time = getTime()
    with torch.no_grad():
        # Detection is handling here
        (img, orig_img, frame, boxes, scores, pt1, pt2) = det_processor.read()

        if orig_img is None:
            return [img, 'rot', 'trans']
        if boxes is None or boxes.nelement() == 0:
            writer.save(None, None, None, None, None, orig_img, frame)
            return [orig_img[0], 'rot', 'trans']

        # Pose Estimation
        img = img.cuda()
        hm =[]
        hm_k = pose_model(img)  # hm is a heatmap with size B*KP*H*W
        hm.append(hm_k)
        hm = torch.cat(hm)
        hm = hm.cpu()  # hm is torch.tensor
        writer.save(boxes, scores, hm, pt1, pt2, orig_img, frame)

    result = writer.read()
    R_p = result['cam_R']
    t_p = result['cam_t']

    if (R_p != []):
        img_with_projection = project_obj_onto_img(orig_img[0], ply_model_corners, R_p, t_p, cam_K)
        return [img_with_projection, R_p, t_p]
    else:
        return [orig_img[0], R_p, t_p]

''' 
    Load cam, model and KP model*******************************************************
'''
class Benchmark:
    def __init__(self):
        self.cam = np.identity(3)
        self.models = {}
        self.kpmodels = {}

def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f)
        return content

def load_sixd_models(base_path, obj_id):
    # This function is used to load sixd benchmark info including camera, model and kp_model.
    print("Loading models and KP models...")
    bench = Benchmark()
    bench.scale_to_meters = 1 # Unit in model is mm
    # You need to give camera info manually here.
    bench.cam = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
    
    #collect model info
    model_info = load_yaml(os.path.join(base_path, 'models', 'models_info.yml'))
    for key, val in model_info.items():
        name = '{:02d}'.format(int(key))
        bench.models[name] = Model3D()
        bench.models[name].diameter = val['diameter']

    # loading models, Linemod has 15 seqs, we use 13(except 3 and 7)
    for ID in range(obj_id, obj_id + 1):
        name = 'obj_{:02d}'.format(ID)
        # embed()
        bench.models['{:02d}'.format(ID)].load(os.path.join(base_path, 'models/' + name + '.ply'), scale=bench.scale_to_meters)
    print("Loading models finished!")

    # loading and refine kp models
    ID = obj_id
    name = 'obj_{:02d}'.format(ID)
    bench.kpmodels['{:02d}'.format(ID)] = Model3D()
    # Modified, take care!
    bench.kpmodels['{:02d}'.format(ID)].load(os.path.join(base_path, 'kpmodels/' + name + '.ply'), scale=bench.scale_to_meters)
    bench.kpmodels['{:02d}'.format(ID)].refine(TOTAL_KP_NUMBER, save=True) # delete too close points

    print("Load and refine KP models finished!")
    return bench

def initialize():
    global image_loader, writer, det_processor, det_loader, pose_model, ply_model_corners, cam_K

    # Loading camera, model, kp_model information of SIXD benchmark datasets
    print ("Betapose begin running now.")
    obj_id = 1
    print("Test seq", obj_id)
    sixd_base = "./betapose/models"
    sixd_bench = load_sixd_models(sixd_base, obj_id)
    cam_K = sixd_bench.cam
    models = sixd_bench.models
    kpmodels = sixd_bench.kpmodels
    ply_model_corners = get_ply_bb_corners(sixd_base + '/models/obj_01.ply')
    kp_model_vertices = kpmodels['{:02d}'.format(int(obj_id))].vertices # used in pnp
    model_vertices = models['{:02d}'.format(int(obj_id))].vertices # used in calculating add

    # Load detection loader
    print('Loading YOLO model..')
    sys.stdout.flush() # for multithread displaying
    image_loader = ImageLoader()
    det_loader = DetectionLoader(image_loader, './betapose/yolo/cfg/yolov3-single.cfg',
                                 './betapose/weights/yolo.weights').start()
    det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model here
    pose_dataset = Mscoco() # is_train, res, joints, rot_factor
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, obj_id, pose_dataset, './betapose/weights/kpd.pkl')
    pose_model.cuda()
    pose_model.eval()

    # Init data writer for writing data and post
    writer = DataWriter(cam_K, 50, kp_model_vertices, args.save_video).start() # save_video default: False

    print('===========================> Finished initialization.')