import glob, os
import shutil
import json
import numpy as np
import random
import yaml
import cv2
from utils import *
from distutils.dir_util import copy_tree

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))


def getBBOfPlyObject(gt_folder, img_num, ply_name):

	with open(gt_folder + '/' + ply_name) as f:
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
		corners = np.array(np.meshgrid(minsMaxs[0,:], minsMaxs[1,:], minsMaxs[2,:])).T.reshape(-1,3)

		corners = np.c_[corners, np.ones((len(corners), 1))].transpose()

		with open(gt_folder + "/gt.yml", 'r') as stream:
			try:
				img_infos = yaml.safe_load(stream)[img_num][0]
				R = np.array(img_infos['cam_R_m2c']).reshape(3, 3)
				Rt = np.append(R, np.array([img_infos['cam_t_m2c']]).T, axis=1)
				np.set_printoptions(suppress=True)
				with open(gt_folder + "/info.yml", 'r') as info_stream:
					try:
						img_infos = yaml.safe_load(info_stream)[img_num]
						i_c = np.array(img_infos['cam_K']).reshape(3, 3)
					except yaml.YAMLError as exc:
						print(exc)
			except yaml.YAMLError as exc:
				print(exc)
			
		proj_2d_p   = compute_projection(corners, Rt, i_c)
		proj_2d_p = proj_2d_p.astype(int)

		# Make empty black image
		image=cv2.imread(gt_folder + '/' + format(img_num, '04') + '.png',1)
		height, width, channels = image.shape
		blue = [255,0,0]
		image[proj_2d_p[1,:],proj_2d_p[0,:]]=blue

		# Draw lower base of 3d bb
		pts = np.array([[proj_2d_p[0,0], proj_2d_p[1,0]],[proj_2d_p[0,2], proj_2d_p[1,2]],[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,1], proj_2d_p[1,1]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))
		
		# Draw the front of the bounding box
		pts = np.array([[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,2], proj_2d_p[1,2]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))

		# Draw upper base of 3d bb
		pts = np.array([[proj_2d_p[0,4], proj_2d_p[1,4]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,5], proj_2d_p[1,5]]], np.int32)
		cv2.polylines(image,[pts],True,(0,255,255))

		# Save
		cv2.imwrite("result.png",image)
		

if __name__ == "__main__":
    # Training settings
	# example: python project_single.py guitar 1499 gibson10x.ply
	folder     = sys.argv[1]
	img_num   = int(sys.argv[2])
	ply_name   = sys.argv[3]
	getBBOfPlyObject(folder, img_num, ply_name)
