# import support libraries
import os
import time
import numpy as np

# import main working libraries
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import sys

# import app libraries
dir_file = os.path.dirname(os.path.realpath(__file__))
print(dir_file)
from .darknet import Darknet
from .utils import *
from .MeshPly import MeshPly

model = None
internal_calibration = None
corners3D = None
test_width = 608  # define test image size
test_height = 608
conf_thresh = 0.1
num_classes = 1

# estimate bounding box
# @torch.no_grad
def initialize_network(datacfg, cfgfile, weightfile):
    # ******************************************#
    #           PARAMETERS PREPARATION          #
    # ******************************************#
    global model, corners3D, internal_calibration

    # parse configuration files
    options = read_data_cfg(datacfg)
    meshname = options['mesh']
    name = options['name']

    # Parameters for the network
    seed = int(time.time())
    gpus = '0'  # define gpus to use
    torch.manual_seed(seed)  # seed torch random
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)  # seed cuda random


    # Read object 3D model, get 3D Bounding box corners
    mesh = MeshPly(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    # print("Vertices are:\n {} Shape: {} Type: {}".format(vertices,vertices.shape, type(vertices)))

    corners3D = get_3D_corners(vertices)
    feet_cm = 30.48  # 1 ft = 30.48 cm
    corners3D[0] = np.array(
        [-11 * feet_cm / 2.0, -11 * feet_cm / 2.0, -11 * feet_cm / 2.0, -11 * feet_cm / 2.0, 11 * feet_cm / 2.0,
         11 * feet_cm / 2.0, 11 * feet_cm / 2.0, 11 * feet_cm / 2.0])
    corners3D[1] = np.array(
        [-feet_cm / 2.0, -feet_cm / 2.0, feet_cm / 2.0, feet_cm / 2.0, -feet_cm / 2.0, -feet_cm / 2.0, feet_cm / 2.0,
         feet_cm / 2.0])
    corners3D[2] = np.array(
        [-11 * feet_cm / 2.0, 11 * feet_cm / 2.0, -11 * feet_cm / 2.0, 11 * feet_cm / 2.0, -11 * feet_cm / 2.0,
         11 * feet_cm / 2.0, -11 * feet_cm / 2.0, 11 * feet_cm / 2.0])
    # print("3D Corners are:\n {} Shape: {} Type: {}".format(corners3D,corners3D.shape, type(corners3D)))

    diam = float(options['diam'])

    # now configure camera intrinsics
    internal_calibration = get_camera_intrinsic()

    # ******************************************#
    #   NETWORK CREATION                        #
    # ******************************************#

    # Create the network based on cfg file
    model = Darknet(cfgfile)
    # model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()


# estimate bounding box
# @torch.no_grad
def evaluate_img(img):
    global model, test_height, test_width, conf_thresh, num_classes


    # Reload Original img
    imgCp = img.copy()
    # ******************************************#
    #   INPUT IMAGE PREPARATION FOR NN          #
    # ******************************************#

    # Now prepare image: convert to RGB, resize, transform to Tensor
    # use cuda,
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    #img = Image.open(imgfile).convert('RGB')
    ori_size = img.size  # store original size
    img = img.resize((test_width, test_height))
    t1 = time.time()
    img = transforms.Compose([transforms.ToTensor(), ])(img)  # .float()
    img = Variable(img, requires_grad=True)
    img = img.unsqueeze(0)
    img = img.cuda()

    # ******************************************#
    #   PASS IT TO NETWORK AND GET PREDICTION   #
    # ******************************************#

    # Forward pass
    output = model(img).data
    # print("Output Size: {}".format(output.size(0)))
    t2 = time.time()

    # ******************************************#
    #       EXTRACT PREDICTIONS                 #
    # ******************************************#

    # Using confidence threshold, eliminate low-confidence predictions
    # and get only boxes over the confidence threshold
    all_boxes = get_region_boxes(output, conf_thresh, num_classes)

    boxes = all_boxes[0]

    # iterate through boxes to find the one with highest confidence
    best_conf_est = -1
    best_box_index = -1
    for j in range(len(boxes)):
        # the confidence is in index = 18
        if (boxes[j][18] > best_conf_est):
            box_pr = boxes[j]  # get bounding box
            best_conf_est = boxes[j][18]
            best_box_index = j
    # print("Best box is: {} and 2D prediction is {}".format(best_box_index,box_pr))
    print("Confidence is: {}".format(best_conf_est))
    if (best_conf_est < conf_thresh):
        return imgCp

    # Denormalize the corner predictions
    # This are the predicted 2D points with which a bounding cube can be drawn
    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * ori_size[0]  # Width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * ori_size[1]  # Height
    t3 = time.time()

    # **********************************************#
    #   GET OBJECT POSE ESTIMATION                  #
    #  Remember the problem in 6D Pose estimation   #
    #  is exactly to estimate the pose - position   #
    #  and orientation of the object of interest    #
    #  with reference to a camera frame. That is    #
    #  why although the 2D projection of the 3D     #
    #  bounding cube are ready, we still need to    #
    #  compute the rotation matrix -orientation-    #
    #  and a translation vector -position- for the  #
    #  object                                       #
    #                                               #
    # **********************************************#

    # get rotation matrix and transform
    R_pr, t_pr = pnp(
        np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),
        corners2D_pr, np.array(internal_calibration, dtype='float32'))
    t4 = time.time()

    # ******************************************#
    #   DISPLAY IMAGE WITH BOUNDING CUBE        #
    # ******************************************#

    # draw each predicted 2D point
    for i, (x, y) in enumerate(corners2D_pr):
        # get colors to draw the lines
        col1 = 28 * i
        col2 = 255 - (28 * i)
        col3 = np.random.randint(0, 256)
        cv2.circle(imgCp, (x, y), 3, (col1, col2, col3), -1)
        cv2.putText(imgCp, str(i), (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

    # Get each predicted point and the centroid
    p1 = corners2D_pr[1]
    p2 = corners2D_pr[2]
    p3 = corners2D_pr[3]
    p4 = corners2D_pr[4]
    p5 = corners2D_pr[5]
    p6 = corners2D_pr[6]
    p7 = corners2D_pr[7]
    p8 = corners2D_pr[8]
    center = corners2D_pr[0]

    # Draw cube lines around detected object
    # draw front face
    line_point = 2
    cv2.line(imgCp, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p2[0], p2[1]), (p4[0], p4[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p4[0], p4[1]), (p3[0], p3[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p3[0], p3[1]), (p1[0], p1[1]), (0, 255, 0), line_point)

    # draw back face
    cv2.line(imgCp, (p5[0], p5[1]), (p6[0], p6[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p7[0], p7[1]), (p8[0], p8[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p6[0], p6[1]), (p8[0], p8[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p5[0], p5[1]), (p7[0], p7[1]), (0, 255, 0), line_point)

    # draw right face
    cv2.line(imgCp, (p2[0], p2[1]), (p6[0], p6[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p1[0], p1[1]), (p5[0], p5[1]), (0, 255, 0), line_point)

    # draw left face
    cv2.line(imgCp, (p3[0], p3[1]), (p7[0], p7[1]), (0, 255, 0), line_point)
    cv2.line(imgCp, (p4[0], p4[1]), (p8[0], p8[1]), (0, 255, 0), line_point)

    # print("Rotation: {}".format(R_pr))
    # print("Translation: {}".format(t_pr))
    # print(" Predict time: {}".format(t2 - t1))
    # print(" 2D Points extraction time: {}".format(t3 - t2))
    # print(" Pose calculation time: {}:".format(t4 - t3))
    # print(" Total time: {}".format(t4 - t1))
    # print("Press any key to close.")

    return imgCp
        