import glob, os
import shutil
import json
import numpy as np
import random
import yaml
import cv2
from utils import *
from distutils.dir_util import copy_tree

w = 640
h = 480

def splitAndCastToFloat(line):
	line = line.split()
	line = line[0:3]
	return list(map(float, line))


def getBBOfPlyObject(gt_folder, ply_name):
	global w, h

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
				with open(gt_folder + "/info.yml", 'r') as info_stream:
					try:
						with open(gt_folder + '/Betapose-results.json') as json_file:
							yaml_gt = yaml.safe_load(stream)
							yaml_info = yaml.safe_load(info_stream)
							data = json.load(json_file)
							print(len(data))
							for result in data:
								img_num = int(result['image_id'].split('.')[0])
								R = np.array(result['cam_R']).reshape(3, 3)
								Rt = np.append(R, np.array([result['cam_t']]).T, axis=1)
								kps = np.array(result['keypoints'])
								bbox = np.array(result['bbox']).astype(int)
								pt_num = int(len(kps)/3)
								kps = kps.reshape(pt_num,3)

								img_infos = yaml_gt[img_num][0]
								R_gt = np.array(img_infos['cam_R_m2c']).reshape(3, 3)
								Rt_gt = np.append(R_gt, np.array([img_infos['cam_t_m2c']]).T, axis=1)
								np.set_printoptions(suppress=True)

								img_infos = yaml_info[img_num]
								i_c = np.array(img_infos['cam_K']).reshape(3, 3)

								proj_2d_gt = compute_projection(corners, Rt_gt, i_c)
								proj_2d_gt = proj_2d_gt.astype(int)


								proj_2d_p = compute_projection(corners, Rt, i_c)
								proj_2d_p = proj_2d_p.astype(int)
								
								# Make empty black image
								image = cv2.imread(gt_folder + '/rgb/' + format(img_num, '04') + '.png', 1)
								height, width, channels = image.shape
								red = [0, 0, 255]
								blue = [255, 0, 0]

								proj_2d_p[1, proj_2d_p[1] < 0] = 0
								proj_2d_p[0, proj_2d_p[0] < 0] = 0
								proj_2d_p[1, proj_2d_p[1] >= h] = h-1
								proj_2d_p[0, proj_2d_p[0] >= w] = w-1

								image[proj_2d_p[1, :], proj_2d_p[0, :]] = blue
								image[proj_2d_gt[1, :], proj_2d_gt[0, :]] = red

								cv2.rectangle(image,(bbox[0], bbox[1]),(bbox[2], bbox[3]),(0,255,0),3)
								for c in kps:
									if (c[2] > 0.2):
										image[int(c[1]), int(c[0])]=(0,255,0)

								# Draw lower base of 3d bb
								pts = np.array([[proj_2d_p[0, 0], proj_2d_p[1, 0]], [proj_2d_p[0, 2], proj_2d_p[1, 2]],
												[proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 1], proj_2d_p[1, 1]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (0, 255, 255))
								pts = np.array([[proj_2d_gt[0, 0], proj_2d_gt[1, 0]], [proj_2d_gt[0, 2], proj_2d_gt[1, 2]],
												[proj_2d_gt[0, 3], proj_2d_gt[1, 3]], [proj_2d_gt[0, 1], proj_2d_gt[1, 1]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (255, 0, 255))

								# Draw the front of the bounding box
								pts = np.array([[proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 7], proj_2d_p[1, 7]],
												[proj_2d_p[0, 6], proj_2d_p[1, 6]], [proj_2d_p[0, 2], proj_2d_p[1, 2]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (0, 255, 255))
								pts = np.array([[proj_2d_gt[0, 3], proj_2d_gt[1, 3]], [proj_2d_gt[0, 7], proj_2d_gt[1, 7]],
												[proj_2d_gt[0, 6], proj_2d_gt[1, 6]], [proj_2d_gt[0, 2], proj_2d_gt[1, 2]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (255, 0, 255))

								# Draw upper base of 3d bb
								pts = np.array([[proj_2d_p[0, 4], proj_2d_p[1, 4]], [proj_2d_p[0, 6], proj_2d_p[1, 6]],
												[proj_2d_p[0, 7], proj_2d_p[1, 7]], [proj_2d_p[0, 5], proj_2d_p[1, 5]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (0, 255, 255))
								pts = np.array([[proj_2d_gt[0, 4], proj_2d_gt[1, 4]], [proj_2d_gt[0, 6], proj_2d_gt[1, 6]],
												[proj_2d_gt[0, 7], proj_2d_gt[1, 7]], [proj_2d_gt[0, 5], proj_2d_gt[1, 5]]],
											   np.int32)
								cv2.polylines(image, [pts], True, (255, 0, 255))

								wname = str(img_num)
								cv2.namedWindow(wname)
								# Show the image and wait key press
								cv2.imshow(wname, image)
								cv2.waitKey()
					except yaml.YAMLError as exc:
						print(exc)
			except yaml.YAMLError as exc:
				print(exc)


		

if __name__ == "__main__":
    # Training settings
	# example: python project_all.py ape_gen obj_01.ply
	folder     = sys.argv[1]
	ply_name   = sys.argv[2]
	getBBOfPlyObject(folder, ply_name)
