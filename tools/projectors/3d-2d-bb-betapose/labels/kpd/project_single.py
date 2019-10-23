import glob, os
import shutil
import json
import numpy as np
import random
import yaml
import cv2
from utils import *
from MeshPly import MeshPly
from distutils.dir_util import copy_tree

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))


def getBBOfPlyObject(gt_folder, img_num, ply_name):
	mesh          = MeshPly(gt_folder + '/' + ply_name)
	vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
	corners3D     = get_3D_corners(vertices)


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
		
	proj_2d_p   = compute_projection(corners3D, Rt, i_c)
	proj_2d_p = proj_2d_p.astype(int)

	proj_2d_vert = compute_projection(vertices, Rt, i_c).astype(int).T


	# Make empty black image
	image=cv2.imread(gt_folder + '/rgb/' + format(img_num, '04') + '.png',1)
	height, width, channels = image.shape
	blue = [255,0,0]
	image[proj_2d_p[1,:],proj_2d_p[0,:]]=blue

	for c in proj_2d_vert:
		if (c[0] < width and c[0] >= 0 and c[1] < height and c[1] >= 0):
			image[c[1], c[0]]=(0,255,0)

	# Draw lower base of 3d bb
	pts = np.array([[proj_2d_p[0,0], proj_2d_p[1,0]],[proj_2d_p[0,2], proj_2d_p[1,2]],[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,1], proj_2d_p[1,1]]], np.int32)
	cv2.polylines(image,[pts],True,(255,0,0))
	
	# Draw the front of the bounding box
	pts = np.array([[proj_2d_p[0,3], proj_2d_p[1,3]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,2], proj_2d_p[1,2]]], np.int32)
	cv2.polylines(image,[pts],True,(0,0,255))

	# Draw upper base of 3d bb
	pts = np.array([[proj_2d_p[0,4], proj_2d_p[1,4]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,7], proj_2d_p[1,7]],[proj_2d_p[0,5], proj_2d_p[1,5]]], np.int32)
	cv2.polylines(image,[pts],True,(0,255,255))

	# Draw -y side
	pts = np.array([[proj_2d_p[0,0], proj_2d_p[1,0]],[proj_2d_p[0,2], proj_2d_p[1,2]],[proj_2d_p[0,6], proj_2d_p[1,6]],[proj_2d_p[0,4], proj_2d_p[1,4]]], np.int32)
	cv2.polylines(image,[pts],True,(0,255,0))

	# Save
	cv2.imwrite("result.png",image)
		

if __name__ == "__main__":
    # Training settings
	# example: python project_single.py guitar 1499 gibson10x.ply
	folder     = sys.argv[1]
	img_num   = int(sys.argv[2])
	ply_name   = sys.argv[3]
	getBBOfPlyObject(folder, img_num, ply_name)
