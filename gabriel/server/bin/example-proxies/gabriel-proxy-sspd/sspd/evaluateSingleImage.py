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

# import app libraries
from darknet import Darknet
from utils import *
from MeshPly import MeshPly


class Line():
    def __init__(self, p1, p2):

        # tilt
        if ((p2[0] - p1[0]) == 0.0):
            self.m = "NaN"  # vertical line
        else:
            self.m = (p2[1] - p1[1]) / (p2[0] - p1[0])

        # intercept
        if (self.m == "NaN"):
            self.b = "NaN"
        else:
            self.b = -1.0 * self.m * p1[0] + p1[1]

        self.p = p1  # store one sample

    def eval(self, x):
        # TODO verify if line is vertical
        return (x * self.m + self.b)


def find_intersection(l1, l2):
    x = (l2.b - l1.b) / (l1.m - l2.m)  # x coord of intersection point
    y = l1.eval(x)  # y coord of intersection point
    return x, y


# estimate bounding box
# @torch.no_grad
def test(datacfg, cfgfile, weightfile, imgfile):
    # ******************************************#
    #           PARAMETERS PREPARATION          #
    # ******************************************#

    # parse configuration files
    options = read_data_cfg(datacfg)
    meshname = options['mesh']
    name = options['name']

    # Parameters for the network
    seed = int(time.time())
    gpus = '0'  # define gpus to use
    test_width = 608  # define test image size
    test_height = 608
    torch.manual_seed(seed)  # seed torch random
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)  # seed cuda random
    conf_thresh = 0.1
    num_classes = 1

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

    # ******************************************#
    #   INPUT IMAGE PREPARATION FOR NN          #
    # ******************************************#

    # Now prepare image: convert to RGB, resize, transform to Tensor
    # use cuda,
    img = Image.open(imgfile).convert('RGB')
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
    # print("Confidence is: {}".format(best_conf_est))
    print(best_conf_est.item(), type(best_conf_est.item()))

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

    # Reload Original img
    img = cv2.imread(imgfile)

    # create a window to display image
    wname = "Prediction"
    cv2.namedWindow(wname)
    # draw each predicted 2D point
    for i, (x, y) in enumerate(corners2D_pr):
        # get colors to draw the lines
        col1 = 28 * i
        col2 = 255 - (28 * i)
        col3 = np.random.randint(0, 256)
        cv2.circle(img, (x, y), 3, (col1, col2, col3), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

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
    cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), line_point)
    cv2.line(img, (p2[0], p2[1]), (p4[0], p4[1]), (0, 255, 0), line_point)
    cv2.line(img, (p4[0], p4[1]), (p3[0], p3[1]), (0, 255, 0), line_point)
    cv2.line(img, (p3[0], p3[1]), (p1[0], p1[1]), (0, 255, 0), line_point)

    # draw back face
    cv2.line(img, (p5[0], p5[1]), (p6[0], p6[1]), (0, 255, 0), line_point)
    cv2.line(img, (p7[0], p7[1]), (p8[0], p8[1]), (0, 255, 0), line_point)
    cv2.line(img, (p6[0], p6[1]), (p8[0], p8[1]), (0, 255, 0), line_point)
    cv2.line(img, (p5[0], p5[1]), (p7[0], p7[1]), (0, 255, 0), line_point)

    # draw right face
    cv2.line(img, (p2[0], p2[1]), (p6[0], p6[1]), (0, 255, 0), line_point)
    cv2.line(img, (p1[0], p1[1]), (p5[0], p5[1]), (0, 255, 0), line_point)

    # draw left face
    cv2.line(img, (p3[0], p3[1]), (p7[0], p7[1]), (0, 255, 0), line_point)
    cv2.line(img, (p4[0], p4[1]), (p8[0], p8[1]), (0, 255, 0), line_point)

    # Calculate gate dimensions
    min_x = np.min(corners3D[0, :])  # this are the gate outermost corners
    max_x = np.max(corners3D[0, :])
    min_y = np.min(corners3D[1, :])
    max_y = np.max(corners3D[1, :])
    min_z = np.min(corners3D[2, :])
    max_z = np.max(corners3D[2, :])

    gate_dim_z = max_z - min_z
    gate_dim_x = max_x - min_x
    gate_dim_y = max_y - min_y

    ############################################################
    #        PREDICT FLYABLE AREA BASED ON ESTIMATED 2D PROJECTIONS
    ############################################################

    # Calculate Fly are based based on offset from predicted 2D
    # Projection
    flyarea_side = 243.84  # cm 8ft
    offset_z = (gate_dim_z - flyarea_side) / 2.0
    offset_x = (gate_dim_x - flyarea_side) / 2.0

    offset_z_ratio = (offset_z / gate_dim_z)  # calculate as ratio wrt side, to use with pixels later
    offset_x_ratio = (offset_x / gate_dim_x)
    # print("Offset X ratio: {}, Offset Z ratio: {}".format(offset_x_ratio,offset_z_ratio))

    #           GATE FRONT
    #
    # array to store all 4 points
    flyarea_corners_front = np.zeros((4, 2), dtype='float32')
    # corner 1
    flyarea_corners_front[0][0] = p4[0] + int((p2[0] - p4[0]) * offset_x_ratio)
    flyarea_corners_front[0][1] = p4[1] + int((p3[1] - p4[1]) * offset_z_ratio)
    # corner 2
    flyarea_corners_front[1][0] = p2[0] + int((p4[0] - p2[0]) * offset_x_ratio)
    flyarea_corners_front[1][1] = p2[1] + int((p1[1] - p2[1]) * offset_x_ratio)
    # corner 3
    flyarea_corners_front[2][0] = p1[0] + int((p3[0] - p1[0]) * offset_x_ratio)
    flyarea_corners_front[2][1] = p1[1] + int((p2[1] - p1[1]) * offset_x_ratio)
    # corner 4
    flyarea_corners_front[3][0] = p3[0] + int((p1[0] - p3[0]) * offset_x_ratio)
    flyarea_corners_front[3][1] = p3[1] + int((p4[1] - p3[1]) * offset_x_ratio)
    # print("Front points: {}".format(flyarea_corners_front))

    # draw front gate area
    fa_p1_f = flyarea_corners_front[0]
    fa_p2_f = flyarea_corners_front[1]
    fa_p3_f = flyarea_corners_front[2]
    fa_p4_f = flyarea_corners_front[3]

    """
    cv2.line(img,(fa_p1_f[0],fa_p1_f[1]),(fa_p2_f[0],fa_p2_f[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p2_f[0],fa_p2_f[1]),(fa_p3_f[0],fa_p3_f[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p4_f[0],fa_p4_f[1]),(fa_p1_f[0],fa_p1_f[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p3_f[0],fa_p3_f[1]),(fa_p4_f[0],fa_p4_f[1]), (255,0,255),line_point)
    """

    #           GATE BACK
    #
    # array to store all 4 points
    flyarea_corners_back = np.zeros((4, 2), dtype='float32')
    # corner 1
    flyarea_corners_back[0][0] = p8[0] + int((p6[0] - p8[0]) * offset_x_ratio)
    flyarea_corners_back[0][1] = p8[1] + int((p7[1] - p8[1]) * offset_z_ratio)
    # corner 2
    flyarea_corners_back[1][0] = p6[0] + int((p8[0] - p6[0]) * offset_x_ratio)
    flyarea_corners_back[1][1] = p6[1] + int((p5[1] - p6[1]) * offset_x_ratio)
    # corner 3
    flyarea_corners_back[2][0] = p5[0] + int((p7[0] - p5[0]) * offset_x_ratio)
    flyarea_corners_back[2][1] = p5[1] + int((p6[1] - p5[1]) * offset_x_ratio)
    # corner 4
    flyarea_corners_back[3][0] = p7[0] + int((p5[0] - p7[0]) * offset_x_ratio)
    flyarea_corners_back[3][1] = p7[1] + int((p8[1] - p7[1]) * offset_x_ratio)
    # print("Back points: {}".format(flyarea_corners_back))

    # draw back gate area
    fa_p1_b = flyarea_corners_back[0]
    fa_p2_b = flyarea_corners_back[1]
    fa_p3_b = flyarea_corners_back[2]
    fa_p4_b = flyarea_corners_back[3]

    """
    cv2.line(img,(fa_p1_b[0],fa_p1_b[1]),(fa_p2_b[0],fa_p2_b[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p2_b[0],fa_p2_b[1]),(fa_p3_b[0],fa_p3_b[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p4_b[0],fa_p4_b[1]),(fa_p1_b[0],fa_p1_b[1]), (255,0,255),line_point)
    cv2.line(img,(fa_p3_b[0],fa_p3_b[1]),(fa_p4_b[0],fa_p4_b[1]), (255,0,255),line_point)
    """

    """
    # draw each predicted 2D point
    for i, (x,y) in enumerate(flyarea_corners_front):
        # get colors to draw the lines
        col1 = 0#np.random.randint(0,256)
        col2 = 0#np.random.randint(0,256)
        col3 = 255#np.random.randint(0,256)
        cv2.circle(img, (x,y), 3, (col1,col2,col3), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)
    # draw each predicted 2D point
    for i, (x,y) in enumerate(flyarea_corners_back):
        # get colors to draw the lines
        col1 = 0#np.random.randint(0,256)
        col2 = 0#np.random.randint(0,256)
        col3 = 255#np.random.randint(0,256)
        cv2.circle(img, (x,y), 3, (col1,col2,col3), -1)
        cv2.putText(img, str(i+4), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)
    """

    #           GATE ALL FRONT AND BACK
    #           LINES
    # FRONT
    front_up = Line(flyarea_corners_front[0], flyarea_corners_front[1])
    front_right = Line(flyarea_corners_front[1], flyarea_corners_front[2])
    front_down = Line(flyarea_corners_front[2], flyarea_corners_front[3])
    front_left = Line(flyarea_corners_front[3], flyarea_corners_front[0])
    # print("Front Up Line: m {:.4f} b{:.4f}".format(front_up.m, front_up.b))
    # print("Front Right Line: m {:.4f} b{:.4f}".format(front_right.m, front_right.b))
    # print("Front Down Line: m {:.4f} b{:.4f}".format(front_down.m, front_down.b))
    # print("Front Left Line: m {:.4f} b{:.4f}".format(front_left.m, front_left.b))

    # BACK
    back_up = Line(flyarea_corners_back[0], flyarea_corners_back[1])
    back_right = Line(flyarea_corners_back[1], flyarea_corners_back[2])
    back_down = Line(flyarea_corners_back[2], flyarea_corners_back[3])
    back_left = Line(flyarea_corners_back[3], flyarea_corners_back[0])
    # print("Back Up Line: m {:.4f} b{:.4f}".format(back_up.m, back_up.b))
    # print("Back Right Line: m {:.4f} b{:.4f}".format(back_right.m, back_right.b))
    # print("Back Down Line: m {:.4f} b{:.4f}".format(back_down.m, back_down.b))
    # print("Back Left Line: m {:.4f} b{:.4f}".format(back_left.m, back_left.b))

    # Intersections
    intersections = np.zeros((8, 2))
    # store in an structure that makes looping easy
    front_lines = [[front_right, front_left], [front_right, front_left], [front_up, front_down], [front_up, front_down]]
    back_lines = [back_up, back_down, back_right, back_left]

    # compare back line with corresponding front lines
    for k, (back_line, front_line_pair) in enumerate(zip(back_lines, front_lines)):
        for j, front_line in enumerate(front_line_pair):
            x_i = (back_line.b - front_line.b) / (front_line.m - back_line.m)  # x coord of intersection point
            y_i = back_line.eval(x_i)  # y coord of intersection point
            intersections[k * 2 + j][0] = x_i
            intersections[k * 2 + j][1] = y_i

    # print("Intersections: ")
    # print(intersections)

    # draw each intersection point
    # for i, (x,y) in enumerate(intersections):
    # cv2.circle(img, (int(x),int(y)), 3, (0,255,255), -1)
    # cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # group all points
    points = np.concatenate((flyarea_corners_front, flyarea_corners_back, intersections), axis=0)

    # the corners of the flyable area is composed of the 4 points with the
    # shortest distance to the centroid
    points_sorted = [(np.linalg.norm(points[i] - center), points[i]) for i in range(points.shape[0])]
    points_sorted.sort()
    # print(points_sorted)

    flyarea_corners = np.zeros((4, 2), dtype='float32')

    """
    for k in range(4):
        #print(k)
        point = points_sorted[k][1]
        #print(point)
        flyarea_corners[k] = point
        x = point[0]
        y = point[1]
        cv2.circle(img, (int(x),int(y)), 10, (0,255,255), -1)
        cv2.putText(img, str(k), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)         
    """

    # corner 1
    x1, y1 = find_intersection(front_up, back_left)
    dummy1 = np.array([x1, y1])
    x1, y1 = find_intersection(back_up, front_left)
    dummy2 = np.array([x1, y1])
    c_points = np.stack((flyarea_corners_front[0], flyarea_corners_back[0], dummy1, dummy2))
    points_sorted = [(np.linalg.norm(c_points[i] - center), c_points[i]) for i in range(c_points.shape[0])]
    points_sorted.sort()
    flyarea_corners[0] = points_sorted[0][1]  # extract the point with shortest distance to centroid

    """
    # draw each intersection point
    for i, (x,y) in enumerate(c_points):
        cv2.circle(img, (int(x),int(y)), 3, (0,255,255), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1) 
    """

    # corner 2
    x1, y1 = find_intersection(front_up, back_right)
    dummy1 = np.array([x1, y1])
    x1, y1 = find_intersection(back_up, front_right)
    dummy2 = np.array([x1, y1])
    c_points = np.stack((flyarea_corners_front[1], flyarea_corners_back[1], dummy1, dummy2))
    points_sorted = [(np.linalg.norm(c_points[i] - center), c_points[i]) for i in range(c_points.shape[0])]
    points_sorted.sort()
    flyarea_corners[1] = points_sorted[0][1]  # extract the point with shortest distance to centroid

    """
    # draw each intersection point
    for i, (x,y) in enumerate(c_points):
        cv2.circle(img, (int(x),int(y)), 3, (0,255,255), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1) 
    """

    # corner 3
    x1, y1 = find_intersection(front_down, back_right)
    dummy1 = np.array([x1, y1])
    x1, y1 = find_intersection(back_down, front_right)
    dummy2 = np.array([x1, y1])
    c_points = np.stack((flyarea_corners_front[2], flyarea_corners_back[2], dummy1, dummy2))
    points_sorted = [(np.linalg.norm(c_points[i] - center), c_points[i]) for i in range(c_points.shape[0])]
    points_sorted.sort()
    flyarea_corners[2] = points_sorted[0][1]  # extract the point with shortest distance to centroid

    """
    # draw each intersection point
    for i, (x,y) in enumerate(c_points):
        cv2.circle(img, (int(x),int(y)), 3, (0,255,255), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1) 
    """

    # corner 4
    x1, y1 = find_intersection(front_down, back_left)
    dummy1 = np.array([x1, y1])
    x1, y1 = find_intersection(back_down, front_left)
    dummy2 = np.array([x1, y1])
    c_points = np.stack((flyarea_corners_front[3], flyarea_corners_back[3], dummy1, dummy2))
    points_sorted = [(np.linalg.norm(c_points[i] - center), c_points[i]) for i in range(c_points.shape[0])]
    points_sorted.sort()
    flyarea_corners[3] = points_sorted[0][1]  # extract the point with shortest distance to centroid

    """
    # draw each intersection point
    for i, (x,y) in enumerate(c_points):
        cv2.circle(img, (int(x),int(y)), 3, (0,255,255), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1) 
    """

    fa_p1 = flyarea_corners[0]
    fa_p2 = flyarea_corners[1]
    fa_p3 = flyarea_corners[2]
    fa_p4 = flyarea_corners[3]

    """
    cv2.line(img,(fa_p1[0],fa_p1[1]),(fa_p2[0],fa_p2[1]), (0,0,255),line_point)
    cv2.line(img,(fa_p2[0],fa_p2[1]),(fa_p3[0],fa_p3[1]), (0,0,255),line_point)
    cv2.line(img,(fa_p4[0],fa_p4[1]),(fa_p1[0],fa_p1[1]), (0,0,255),line_point)
    cv2.line(img,(fa_p3[0],fa_p3[1]),(fa_p4[0],fa_p4[1]), (0,0,255),line_point)
    """

    #     YET ANOTHER METHOD
    if (back_up.p[1] > front_up.p[1]):
        up_line = back_up
    else:
        up_line = front_up
    if (back_down.p[1] < front_down.p[1]):
        down_line = back_down
    else:
        down_line = front_down
    if (back_right.p[0] < front_right.p[0]):
        right_line = back_right
    else:
        right_line = front_right
    if (back_left.p[0] > front_left.p[0]):
        left_line = back_left
    else:
        left_line = front_left

    x1, y1 = find_intersection(up_line, left_line)
    dummy1 = np.array([x1, y1])
    flyarea_corners[0] = dummy1
    x1, y1 = find_intersection(up_line, right_line)
    dummy1 = np.array([x1, y1])
    flyarea_corners[1] = dummy1
    x1, y1 = find_intersection(down_line, right_line)
    dummy1 = np.array([x1, y1])
    flyarea_corners[2] = dummy1
    x1, y1 = find_intersection(down_line, left_line)
    dummy1 = np.array([x1, y1])
    flyarea_corners[3] = dummy1

    fa_p1 = flyarea_corners[0]
    fa_p2 = flyarea_corners[1]
    fa_p3 = flyarea_corners[2]
    fa_p4 = flyarea_corners[3]

    cv2.line(img, (fa_p1[0], fa_p1[1]), (fa_p2[0], fa_p2[1]), (0, 0, 255), line_point)
    cv2.line(img, (fa_p2[0], fa_p2[1]), (fa_p3[0], fa_p3[1]), (0, 0, 255), line_point)
    cv2.line(img, (fa_p4[0], fa_p4[1]), (fa_p1[0], fa_p1[1]), (0, 0, 255), line_point)
    cv2.line(img, (fa_p3[0], fa_p3[1]), (fa_p4[0], fa_p4[1]), (0, 0, 255), line_point)

    """
    ############################################################
    #        PREDICT FLYABLE AREA BASED ON ESTIMATED POSE
    ############################################################
    offset = 0.0   # flyable area corners are at an offset from outermost corners
    y = min_y       # and they are over a plane
    p1 = np.array([[min_x+offset],[y],[min_z+offset]])
    p2 = np.array([[min_x+offset],[y],[max_z-offset]])
    p3 = np.array([[max_x-offset],[y],[min_z+offset]])
    p4 = np.array([[max_x-offset],[y],[max_z-offset]])
    # These are 4 points defining the square of the flyable area
    flyarea_3Dpoints = np.concatenate((p1,p2,p3,p4), axis = 1)
    flyarea_3Dpoints = np.concatenate((flyarea_3Dpoints, np.ones((1,4))), axis = 0)
    print("Gate Flyable Area 3D:\n{}".format(flyarea_3Dpoints))
    # get transform
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1) 
    flyarea_2Dpoints = compute_projection(flyarea_3Dpoints, Rt_pr, internal_calibration)
    print("Gate Flyable Area 2D projection:\n{}".format(flyarea_2Dpoints))
    for i,(x,y) in enumerate(flyarea_2Dpoints.T):
        col1 = 0#np.random.randint(0,256)
        col2 = 0#np.random.randint(0,256)
        col3 = 255#np.random.randint(0,256)
        cv2.circle(img, (x,y), 3, (col1,col2,col3), -1)
        cv2.putText(img, str(i), (int(x) + 5, int(y) + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)        
    p1_2d = np.array([ flyarea_2Dpoints[0][0], flyarea_2Dpoints[1][0]])
    p2_2d = np.array([ flyarea_2Dpoints[0][1], flyarea_2Dpoints[1][1]])
    p3_2d = np.array([ flyarea_2Dpoints[0][2], flyarea_2Dpoints[1][2]])
    p4_2d = np.array([ flyarea_2Dpoints[0][3], flyarea_2Dpoints[1][3]])
    """

    # Show the image and wait key press
    cv2.imshow(wname, img)
    cv2.waitKey()

    print("Rotation: {}".format(R_pr))
    print("Translation: {}".format(t_pr))
    print(" Predict time: {}".format(t2 - t1))
    print(" 2D Points extraction time: {}".format(t3 - t2))
    print(" Pose calculation time: {}:".format(t4 - t3))
    print(" Total time: {}".format(t4 - t1))
    print("Press any key to close.")


if __name__ == '__main__':
    import sys

    if (len(sys.argv) == 5):
        datacfg_file = sys.argv[1]  # data file
        cfgfile_file = sys.argv[2]  # yolo network file
        weightfile_file = sys.argv[3]  # weightd file
        imgfile_file = sys.argv[4]  # image file
        test(datacfg_file, cfgfile_file, weightfile_file, imgfile_file)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile imagefile')