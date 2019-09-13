import glob, os
import shutil
import json
import numpy as np
import random
import cv2
from .utils import *
from distutils.dir_util import copy_tree

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))

def get_ply_bb_corners(ply_model):
	with open(ply_model) as f:
		content = f.readlines()
		content = [x.strip() for x in content]

		skip = 0
		foundVertexEl = False
		foundEndOfHead = False
		lineVals = []
		linesToScan = 0

		while (not foundVertexEl or not foundEndOfHead):
			lineVals = content[skip].split()
			if (lineVals[0] == 'end_header'):
				foundEndOfHead = True
			if (lineVals[0] == 'element'):
				if (lineVals[1] == 'vertex'):
					linesToScan = int(lineVals[2])
					foundVertexEl = True
			skip += 1
		content = content[skip:linesToScan + skip]
		copy = []
		for line in content:
			copy.append(splitAndCastToFloat(line))
		vertices = np.matrix(np.array(copy))
		mins = vertices.min(0)
		maxs = vertices.max(0)
		minsMaxs = np.array([[mins.item(0), mins.item(1), mins.item(2)], [maxs.item(0), maxs.item(1), maxs.item(2)]]).T
		corners = np.array(np.meshgrid(minsMaxs[0, :], minsMaxs[1, :], minsMaxs[2, :])).T.reshape(-1, 3)
		return corners

def draw_3d_bounding_box(imgCp, corners2D_pr, bb_3d_color, line_point):
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
    cv2.line(imgCp, (p1[0], p1[1]), (p2[0], p2[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p2[0], p2[1]), (p4[0], p4[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p4[0], p4[1]), (p3[0], p3[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p3[0], p3[1]), (p1[0], p1[1]), bb_3d_color, line_point)

    # draw back face
    cv2.line(imgCp, (p5[0], p5[1]), (p6[0], p6[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p7[0], p7[1]), (p8[0], p8[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p6[0], p6[1]), (p8[0], p8[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p5[0], p5[1]), (p7[0], p7[1]), bb_3d_color, line_point)

    # draw right face
    cv2.line(imgCp, (p2[0], p2[1]), (p6[0], p6[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p1[0], p1[1]), (p5[0], p5[1]), bb_3d_color, line_point)

    # draw left face
    cv2.line(imgCp, (p3[0], p3[1]), (p7[0], p7[1]), bb_3d_color, line_point)
    cv2.line(imgCp, (p4[0], p4[1]), (p8[0], p8[1]), bb_3d_color, line_point)

    return imgCp	

def project_obj_onto_img(image, corners, R_p, t_p, i_c, bbs, kps, score):
	if (R_p != []):
		corners = np.c_[corners, np.ones((len(corners), 1))].transpose()

		Rt = np.append(R_p, t_p, axis=1)
		proj_2d_p   = compute_projection(corners, Rt, i_c)
		proj_2d_p = proj_2d_p.astype(int)

		proj_2d_p[1,proj_2d_p[1] >= 480] = 479
		proj_2d_p[0,proj_2d_p[0] >= 640] = 639

		proj_2d_p[1,proj_2d_p[1] < 0] = 0
		proj_2d_p[0,proj_2d_p[0] < 0] = 0

		# Draw all KPs
		for c in kps:
			cv2.circle(image, (int(c[0]), int(c[1])), 2, (255, 0, 255), -1)

		# Draw bounding box resulting after PNP
		sum_x = np.mean(proj_2d_p[0])
		sum_y = np.mean(proj_2d_p[1])
		proj_2d_p = np.concatenate((np.array([[sum_x,sum_y]]), proj_2d_p.T), axis=0).astype(int)
		image = draw_3d_bounding_box(image, proj_2d_p, (0,255,0), 3)

	# Draw YOLO bounding box
	for bb in bbs:
		bb[bb < 0] = 0
		cv2.rectangle(image, (bb[0], bb[1]),
					  (bb[2], bb[3]), (255, 0, 0), 1)

	cv2.putText(image, f'{score:.3f}', (640 - 50, 480 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
	return image
