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

	image_output_folder = './image_output2/'
	image_input_folder = './images_input2/'
	betapose_results = './results/betapose/tad/Betapose-results2.json'
	sspe_results = './results/sspe/tad/pr2/'

	#i_c = np.array([565,0,320,0,605,240,0,0,1]).reshape(3, 3)
	#i_c = np.array([320., 0.0, 320., 0.0, 320, 240., 0.0, 0.0, 1.0],dtype=float).reshape(3, 3)
	i_c = np.array([320., 0.0, 320., 0.0, 320, 240., 0.0, 0.0, 1.0],dtype=float).reshape(3, 3)

	line_width = 2
	color_pr_sspe = (0,255,0)
	color_pr_betapose = (0,0,255)
	color_gt = (255,0,0)

	if not os.path.exists(image_output_folder):
		os.makedirs(image_output_folder)

	for file in sorted(os.listdir(image_input_folder)):
		shutil.copy(os.path.join(image_input_folder, file), os.path.join(image_output_folder, file))


	# load ply model
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

		# Load the results for betapose and sspe for each image, project it and save it
		with open(betapose_results) as json_file:
			data = json.load(json_file)

			for file in sorted(os.listdir(image_input_folder)):
				target_img = int(file.split('.')[0])
				proj_2d_p_sspe = np.array([])
				if os.path.isfile(sspe_results + 'corners_' + format(target_img, '04') + '.txt'):
					R_sspe = np.loadtxt(sspe_results + 'R_' +  format(target_img, '04') + '.txt')
					t_sspe = np.array([np.loadtxt(sspe_results + 't_' +  format(target_img, '04') + '.txt')]).T
					Rt_sspe = np.append(R_sspe, t_sspe, axis=1)
					proj_2d_p_sspe = compute_projection(corners, Rt_sspe, i_c)
					proj_2d_p_sspe = proj_2d_p_sspe.astype(int)
					proj_2d_p_sspe[1, proj_2d_p_sspe[1] < 0] = 0
					proj_2d_p_sspe[0, proj_2d_p_sspe[0] < 0] = 0
					proj_2d_p_sspe[1, proj_2d_p_sspe[1] >= h] = h-1
					proj_2d_p_sspe[0, proj_2d_p_sspe[0] >= w] = w-1
					sum_x = np.mean(proj_2d_p_sspe[0])
					sum_y = np.mean(proj_2d_p_sspe[1])
					proj_2d_p_sspe = np.concatenate((np.array([[sum_x,sum_y]]), proj_2d_p_sspe.T), axis=0).astype(int).T
				for result in data:
					img_num = int(result['image_id'].split('.')[0])
					if (img_num != target_img):
						continue
					# Make empty black image
					image = cv2.imread(image_input_folder + file, 1)
					height, width, channels = image.shape
					red = [0, 0, 255]
					blue = [255, 0, 0]

					if ('cam_R' in result):
						R = np.array(result['cam_R']).reshape(3, 3)
						Rt = np.append(R, np.array([result['cam_t']]).T, axis=1)
						proj_2d_p_beta = compute_projection(corners, Rt, i_c)
						proj_2d_p_beta = proj_2d_p_beta.astype(int)
						proj_2d_p_beta[1, proj_2d_p_beta[1] < 0] = 0
						proj_2d_p_beta[0, proj_2d_p_beta[0] < 0] = 0
						proj_2d_p_beta[1, proj_2d_p_beta[1] >= h] = h-1
						proj_2d_p_beta[0, proj_2d_p_beta[0] >= w] = w-1
						sum_x = np.mean(proj_2d_p_beta[0])
						sum_y = np.mean(proj_2d_p_beta[1])
						proj_2d_p_beta = np.concatenate((np.array([[sum_x,sum_y]]), proj_2d_p_beta.T), axis=0).astype(int).T
						image = draw_bb(image, proj_2d_p_beta, color_pr_betapose, line_width)

					if proj_2d_p_sspe.shape != (0,):
						image = draw_bb(image, proj_2d_p_sspe, color_pr_sspe, line_width)

					cv2.imwrite(image_output_folder + file,image)


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