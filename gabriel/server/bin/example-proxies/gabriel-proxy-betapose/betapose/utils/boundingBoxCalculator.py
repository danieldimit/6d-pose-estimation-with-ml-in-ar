import glob, os
import shutil
import json
import numpy as np
import random
import cv2
from . import *
from distutils.dir_util import copy_tree

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))


def getBBOfPlyObject():
	with open("ape.ply") as f:
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
			skip+=1
		content = content[skip:linesToScan+skip]
		copy = [];
		for line in content: 
			copy.append(splitAndCastToFloat(line))
		vertices = np.matrix(np.array(copy))
		mins = vertices.min(0)
		maxs = vertices.max(0)
		minsMaxs = np.array([[mins.item(0),mins.item(1),mins.item(2)], [maxs.item(0),maxs.item(1),maxs.item(2)]]).T
		print(minsMaxs)
		corners = np.array(np.meshgrid(minsMaxs[0,:], minsMaxs[1,:], minsMaxs[2,:])).T.reshape(-1,3)
		R_gt = np.array([[0.94137597, 0.31805399, -0.11248200, 11.03801096], [0.23415700, -0.85603601, -0.46083900, 124.58541745], [-0.24286000, 0.40748399, -0.88032699, 849.71766750]])
		R_p = np.array([[0.9406752291304329, 0.3258326502956354, -0.09467416385545607, 11.76421530495168], [0.270510546808252, -0.8885950780502103, -0.37043600166620033, 127.83885882665334], [-0.20482714020829385, 0.3228496109123609, -0.9240205470485108, 866.1953181788415]])
		t_gt = np.array([11.76421530495168, 127.83885882665334, 866.1953181788415])
		#Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
		i_c = np.array([[572.4114, 0. ,325.2611], [  0.   ,  573.5704, 242.0489], [  0. ,      0. ,      1.    ]])
		
		corners = np.c_[corners, np.ones((len(corners), 1))].transpose()
		print(corners)
		print(R_gt)
		print(i_c)
		proj_2d_gt   = compute_projection(corners, R_gt, i_c)
		proj_2d_gt = proj_2d_gt.astype(int)

		with open('res.json') as json_file:  
			data = json.load(json_file)
			R = np.array(data[0]['cam_R']).reshape(3,3)
			Rt = np.append(R, np.array([data[0]['cam_t']]).T, axis=1)
		proj_2d_p   = compute_projection(corners, Rt, i_c)
		proj_2d_p = proj_2d_p.astype(int)
		print(proj_2d_gt)
		print(proj_2d_p)
		# Make empty black image
		image=cv2.imread('000000000505.png',1)
		height, width, channels = image.shape
		red = [0,0,255]
		blue = [255,0,0]
		image[proj_2d_p[1,:],proj_2d_p[0,:]]=blue
		image[proj_2d_gt[1,:],proj_2d_gt[0,:]]=red

		# Draw lower base of 3d bb
		pts = np.array([[proj_2d_p[0,0], proj_2d_p[1,0]],[proj_2d_p[0,2], proj_2d_p[1,2]],[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,1], proj_2d_p[1,1]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))
		pts = np.array([[proj_2d_gt[0,0], proj_2d_gt[1,0]],[proj_2d_gt[0,2], proj_2d_gt[1,2]],[proj_2d_gt[0,3], proj_2d_gt[1,3]],[proj_2d_gt[0,1], proj_2d_gt[1,1]]], np.int32)
		cv2.polylines(image,[pts],True,(255,0,255))
		
		# Draw the front of the bounding box
		pts = np.array([[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,2], proj_2d_p[1,2]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))
		pts = np.array([[proj_2d_gt[0,3], proj_2d_gt[1,3]],[proj_2d_gt[0,7], proj_2d_gt[1,7]],[proj_2d_gt[0,6], proj_2d_gt[1,6]],[proj_2d_gt[0,2], proj_2d_gt[1,2]]], np.int32)
		cv2.polylines(image,[pts],True,(255,0,255))

		# Draw upper base of 3d bb
		pts = np.array([[proj_2d_p[0,4], proj_2d_p[1,4]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,5], proj_2d_p[1,5]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))
		pts = np.array([[proj_2d_gt[0,4], proj_2d_gt[1,4]],[proj_2d_gt[0,6], proj_2d_gt[1,6]],[proj_2d_gt[0,7], proj_2d_gt[1,7]],[proj_2d_gt[0,5], proj_2d_gt[1,5]]], np.int32)
		cv2.polylines(image,[pts],True,(255,0,255))

		# Save
		cv2.imwrite("result.png",image)
		

getBBOfPlyObject()
