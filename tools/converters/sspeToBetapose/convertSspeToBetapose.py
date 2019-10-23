import glob, os, sys
import json
import numpy as np
import cv2
import math
import shutil
from utils import *
from MeshPly import MeshPly

output = 'betapose'
sspe = 'sspe'
ply_name = 'obj_01.ply'
width = 640
height = 480
mesh          = MeshPly(ply_name)
vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D     = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), get_3D_corners(vertices)[:3, :]), axis=1)), dtype='float32')

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    while (points_2D.shape[0] <= 5):
        points_2D = np.append(points_2D, [points_2D[points_2D.shape[0]-1]], axis=0)
        points_3D = np.append(points_3D, [points_3D[points_3D.shape[0]-1]], axis=0)

    _, R_exp, t = cv2.solvePnP(points_3D,
                              # points_2D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)

    R, _ = cv2.Rodrigues(R_exp)
    return (R, t)



def extractPointsFromSspeLabel(file_name):
	with open(file_name) as lb_f:
		labels = lb_f.readline()
	labels = labels.split()

	labels_float = []
	min_x = 99999999999999
	min_y = 99999999999999
	max_x = -99999999999999
	max_y = -99999999999999

	i = -1
	for label in labels:
		i+=1
		if (i == 0):
			obj_id = str(int(label) + 1)
			continue
		if (i == 19):
			x_range = int(float(label) * width)
			continue
		if (i == 20):
			y_range = int(float(label) * height)
			continue

		if (i % 2 == 0):
			y = float(label) * height
			if (min_y > y):
				min_y = int(y)
			if (max_y < y):
				max_y = int(y)
			labels_float.append(float(label) * height)
		else:
			x = float(label) * width
			if (min_x > x):
				min_x = int(x)
			if (max_x < x):
				max_x = int(x)
			labels_float.append(x)
		
	return min_x, min_y, max_x, max_y, obj_id, np.array(labels_float).reshape(9,2)

def createLabelContent():

	if not os.path.exists(output):
		os.makedirs(output)
		os.makedirs(output + '/rgb')

	f = open(output + '/gt.yml', "w+")
	f_c = open(output + '/info.yml', "w+")

	for file in sorted(os.listdir(sspe + '/labels/')):
		index = int(file.split('.')[0])

		image_adr = sspe + '/JPEGImages/' + file.split('.')[0] + '.jpg'
		label_adr = sspe + '/labels/' + file

		# copy image
		file_name = format(index, '04') + '.jpg'
		shutil.copy(image_adr, os.path.join(output + '/rgb', file_name))

		# compute R and t via pnp
		min_x, min_y, max_x, max_y, obj_id, marked_points = extractPointsFromSspeLabel(label_adr)
		i_c = np.array([565,0,320,0,605,240,0,0,1]).reshape(3,3).astype(float)
		R, t = pnp(corners3D, marked_points, np.array(i_c, dtype='float32'))
		f.write(str(index) + ':\n')
		f.write('- cam_R_m2c: ' + str(R.flatten().tolist()) + '\n')
		f.write('  cam_t_m2c: ' + str(t.flatten().tolist()) + '\n')
		f.write('  obj_bb: ' + str([min_x, min_y, max_x - min_x, max_y - min_y]) + '\n')
		f.write('  obj_id: ' + obj_id + '\n')

		# write cam info
		f_c.write(str(index) + ':\n')
		f_c.write('  cam_K: [565,0,320,0,605,240,0,0,1]\n')
		f_c.write('  depth_scale: 1.0\n')

	f.close()
	f_c.close()






	
if __name__ == "__main__":
	createLabelContent()

