
import glob, os
import shutil
import json
import numpy as np
import random
import cv2
from distutils.dir_util import copy_tree

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))


def getBBOfPlyObject():
	proj_2d_gt = np.loadtxt('corners_0505_gt.txt')
	proj_2d_p = np.loadtxt('corners_0505_pr.txt')
	proj_2d_gt = proj_2d_gt.astype(int).T
	proj_2d_p = proj_2d_p.astype(int).T
	print(proj_2d_gt)
	print(proj_2d_p)
	# Make empty black image
	image=cv2.imread('000505.jpg',1)
	height, width, channels = image.shape
	
	color_pr = (0,255,255)
	color_gt = (255,0,255)
	image[proj_2d_p[1,:],proj_2d_p[0,:]]=color_pr
	image[proj_2d_gt[1,:],proj_2d_gt[0,:]]=color_gt


	# Draw the front of the bounding box
	pts = np.array([[proj_2d_p[0,1], proj_2d_p[1,1]],[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,4], proj_2d_p[1,4]],[proj_2d_p[0,2], proj_2d_p[1,2]]], np.int32)
	cv2.polylines(image,[pts],True,color_pr)
	pts = np.array([[proj_2d_gt[0,1], proj_2d_gt[1,1]],[proj_2d_gt[0,3], proj_2d_gt[1,3]],[proj_2d_gt[0,4], proj_2d_gt[1,4]],[proj_2d_gt[0,2], proj_2d_gt[1,2]]], np.int32)
	cv2.polylines(image,[pts],True,color_gt)
	
	# Draw lower base of 3d bb
	pts = np.array([[proj_2d_p[0,1], proj_2d_p[1,1]],[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,5], proj_2d_p[1,5]]], np.int32)
	cv2.polylines(image,[pts],True,color_pr)
	pts = np.array([[proj_2d_gt[0,1], proj_2d_gt[1,1]],[proj_2d_gt[0,3], proj_2d_gt[1,3]],[proj_2d_gt[0,7], proj_2d_gt[1,7]],[proj_2d_gt[0,5], proj_2d_gt[1,5]]], np.int32)
	cv2.polylines(image,[pts],True,color_gt)

	# Draw upper base of 3d bb
	pts = np.array([[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,2], proj_2d_p[1,2]],[proj_2d_p[0,4], proj_2d_p[1,4]],[proj_2d_p[0,8], proj_2d_p[1,8]]], np.int32)
	cv2.polylines(image,[pts],True,color_pr)
	pts = np.array([[proj_2d_gt[0,6], proj_2d_gt[1,6]],[proj_2d_gt[0,2], proj_2d_gt[1,2]],[proj_2d_gt[0,4], proj_2d_gt[1,4]],[proj_2d_gt[0,8], proj_2d_gt[1,8]]], np.int32)
	cv2.polylines(image,[pts],True,color_gt)

	# Save
	cv2.imwrite("result.png",image)
		

getBBOfPlyObject()
