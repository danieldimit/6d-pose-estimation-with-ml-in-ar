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


def printBB():
	global w, h

	target_img = 34
	gt_folder_beta = './results/betapose/real_full'

	proj_2d_p_sspe = np.loadtxt('./results/sspe/real_full/pr/corners_' + format(target_img, '04') + '.txt')

	proj_2d_p_sspe = proj_2d_p_sspe.astype(int).T




	line_width = 2

	color_pr_sspe = (0,255,0)
	color_pr_betapose = (0,0,255)
	color_gt = (255,0,0)

	with open('kuka.ply') as f:
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

		with open(gt_folder_beta + "/gt.yml", 'r') as stream:
			try:
				with open(gt_folder_beta + "/info.yml", 'r') as info_stream:
					try:
						with open(gt_folder_beta + '/Betapose-results.json') as json_file:
							yaml_gt = yaml.safe_load(stream)
							yaml_info = yaml.safe_load(info_stream)
							data = json.load(json_file)
							print(len(data))
							for result in data:
								img_num = int(result['image_id'].split('.')[0])
								if (img_num != target_img):
									continue
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
								image = cv2.imread(gt_folder_beta + '/rgb/' + format(img_num, '04') + '.jpg', 1)
								height, width, channels = image.shape
								red = [0, 0, 255]
								blue = [255, 0, 0]

								proj_2d_p[1, proj_2d_p[1] < 0] = 0
								proj_2d_p[0, proj_2d_p[0] < 0] = 0
								proj_2d_p[1, proj_2d_p[1] >= h] = h-1
								proj_2d_p[0, proj_2d_p[0] >= w] = w-1

								sum_x = np.mean(proj_2d_p[0])
								sum_y = np.mean(proj_2d_p[1])
								proj_2d_p = np.concatenate((np.array([[sum_x,sum_y]]), proj_2d_p.T), axis=0).astype(int).T

								
								image = draw_bb(image, proj_2d_gt_sspe, color_gt, line_width)
								image = draw_bb(image, proj_2d_p_sspe, color_pr_sspe, line_width)
								image = draw_bb(image, proj_2d_p, color_pr_betapose, line_width)



								wname = str(img_num)
								cv2.namedWindow(wname)
								# Show the image and wait key press
								cv2.imshow(wname, image)
								cv2.waitKey()
								cv2.imwrite("result.png",image)
					except yaml.YAMLError as exc:
						print(exc)
			except yaml.YAMLError as exc:
				print(exc)


def draw_bb(imgCp, corners2D_pr, bb_3d_color, line_point):
	overlay = imgCp.copy()
	alpha = 0.3

	corners2D_pr = corners2D_pr.T
	p1 = corners2D_pr[1]
	p2 = corners2D_pr[2]
	p3 = corners2D_pr[3]
	p4 = corners2D_pr[4]
	p5 = corners2D_pr[5]
	p6 = corners2D_pr[6]
	p7 = corners2D_pr[7]
	p8 = corners2D_pr[8]
	center = corners2D_pr[0]

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

	imgCp = cv2.addWeighted(overlay, alpha, imgCp, 1 - alpha, 0)
	return imgCp
		

if __name__ == "__main__":
    # Training settings
	# example: python project_all.py ape_gen obj_01.ply
	printBB()
