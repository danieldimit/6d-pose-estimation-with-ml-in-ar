from utils import *
from MeshPly import MeshPly
import numpy as np
from matrixUtils import *

class BoundingBox:


	def __init__(self, mesh_name):
		self.i_c = get_camera_intrinsic()
		# Read object model information, get 3D bounding box corners
		self.mesh          = MeshPly(mesh_name)
		self.vertices      = np.c_[np.array(self.mesh.vertices), np.ones((len(self.mesh.vertices), 1))].transpose()
		self.corners3D     = get_3D_corners(self.vertices)
		# diam          = calc_pts_diameter(np.array(mesh.vertices))
		self.diam          = float(0.6536639)
		self.w = 640
		self.h = 480
		self.marked_points = []

	def add_clicked_point(self, point):
		self.marked_points.append(point)
		if (len(self.marked_points) == 8):
			self.marked_points = np.asarray(self.marked_points, dtype=float).T
			self.marked_points[0] = self.marked_points[0]
			self.marked_points[1] = self.marked_points[1]
			all_points = np.array(np.transpose(self.corners3D[:3, :]), dtype='float32')
			lower_points = np.array([all_points[0], all_points[2], all_points[4], all_points[6]])
			R, t = pnp(all_points, self.marked_points.T, np.array(self.i_c, dtype='float32'))
			Rt = np.append(R, t, axis=1)
			proj_2d_p = compute_projection(self.corners3D, Rt, self.i_c)
			self.marked_points = []
			return rotationMatrixToEulerAngles(R), t
		return None
		

	def draw_on_img(self, image, Rt):
		image_tmp = image.copy()
		proj_2d_p = compute_projection(self.corners3D, Rt, self.i_c)
		
		proj_2d_p = proj_2d_p.astype(int)

		for c in self.marked_points:
			cv2.circle(image_tmp,(c[0], c[1]), 5, (0,255,0), -1)

		cv2.circle(image_tmp,(proj_2d_p[0,0], proj_2d_p[1,0]), 5, (0,255,0), -1)
		cv2.circle(image_tmp,(proj_2d_p[0,2], proj_2d_p[1,2]), 5, (0,255,0), -1)
		cv2.circle(image_tmp,(proj_2d_p[0,3], proj_2d_p[1,3]), 5, (0,255,0), -1)
		cv2.circle(image_tmp,(proj_2d_p[0,1], proj_2d_p[1,1]), 5, (0,255,0), -1)

		# Draw lower base of 3d bb
		pts = np.array([[proj_2d_p[0, 0], proj_2d_p[1, 0]], [proj_2d_p[0, 2], proj_2d_p[1, 2]],
						[proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 1], proj_2d_p[1, 1]]],
					   np.int32)
		cv2.polylines(image_tmp, [pts], True, (255, 0, 0))

		# Draw the front of the bounding box
		pts = np.array([[proj_2d_p[0, 3], proj_2d_p[1, 3]], [proj_2d_p[0, 7], proj_2d_p[1, 7]],
						[proj_2d_p[0, 6], proj_2d_p[1, 6]], [proj_2d_p[0, 2], proj_2d_p[1, 2]]],
					   np.int32)
		cv2.polylines(image_tmp, [pts], True, (0, 255, 255))

		# Draw upper base of 3d bb
		pts = np.array([[proj_2d_p[0, 4], proj_2d_p[1, 4]], [proj_2d_p[0, 6], proj_2d_p[1, 6]],
						[proj_2d_p[0, 7], proj_2d_p[1, 7]], [proj_2d_p[0, 5], proj_2d_p[1, 5]]],
					   np.int32)
		cv2.polylines(image_tmp, [pts], True, (0, 255, 255))
		return image_tmp