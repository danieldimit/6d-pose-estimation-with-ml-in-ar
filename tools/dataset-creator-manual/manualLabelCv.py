import argparse
import cv2
import os
from utils import *
from matrixUtils import *
from BoundingBox import BoundingBox
 
# Settings
refPt = []
files = []
file_counter = 0
angle_max = 720000000
angle_full_range = 6.4
t_range = 4000
t_grnlrty = 200
image = None
clone = None

# Variables dependent on the settings
t_start = int(t_range / 2)
angle_subdiv = angle_max/angle_full_range

def move_bounding_box(save=False):
	global bb_calc, image, clone, a_x, a_y, a_z, t_x, t_y, t_z
	R = eulerAnglesToRotationMatrix([a_x/(angle_subdiv),a_y/(angle_subdiv),a_z/(angle_subdiv)])
	t = np.array([((t_x-(t_range/2))/t_grnlrty,(t_y-(t_range/2))/t_grnlrty,(t_z-(t_range/2))/t_grnlrty)], dtype=float).T
	image_tmp = clone.copy()
	image = bb_calc.draw_on_img(image_tmp, R, t, save=save)

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, image, clone, a_x, a_y, a_z, t_x, t_y, t_z

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		result = bb_calc.add_clicked_point(np.array([x, y]))
		if (result is not None):
			angles = (result[0] * angle_subdiv).astype(int)
			t = (result[1] * t_grnlrty + t_start).astype(int)
			t_x = t[0][0]
			t_y = t[1][0]
			t_z = t[2][0]
			cv2.setTrackbarPos('t_x', 'image', t_x)
			cv2.setTrackbarPos('t_y', 'image', t_y)
			cv2.setTrackbarPos('t_z', 'image', t_z)

			a_x = angles[0]
			a_y = angles[1]
			a_z = angles[2]
			cv2.setTrackbarPos('R_x', 'image', a_x)
			cv2.setTrackbarPos('R_y', 'image', a_y)
			cv2.setTrackbarPos('R_z', 'image', a_z)
		move_bounding_box()
		
		

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagesFolder", required=True, help="Path to the image")
ap.add_argument("-m", "--mesh", required=True, help="Path to the image")
args = vars(ap.parse_args())


bb_calc = BoundingBox(args["mesh"])

# load the image, clone it, and setup the mouse callback function
files = sorted(os.listdir(args["imagesFolder"]))
if os.path.exists('./labels'):
	file_counter = len(sorted(os.listdir("./labels")))
image = cv2.imread(os.path.join(args["imagesFolder"], files[file_counter]))
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

cv2.createTrackbar('t_x','image',t_start,t_range,move_bounding_box)
cv2.createTrackbar('t_y','image',t_start,t_range,move_bounding_box)
cv2.createTrackbar('t_z','image',t_start,t_range,move_bounding_box)

cv2.createTrackbar('R_x','image',0,angle_max,move_bounding_box)
cv2.createTrackbar('R_y','image',0,angle_max,move_bounding_box)
cv2.createTrackbar('R_z','image',0,angle_max,move_bounding_box)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	t_x = cv2.getTrackbarPos('t_x','image')
	t_y = cv2.getTrackbarPos('t_y','image')
	t_z = cv2.getTrackbarPos('t_z','image')

	a_x = cv2.getTrackbarPos('R_x','image')
	a_y = cv2.getTrackbarPos('R_y','image')
	a_z = cv2.getTrackbarPos('R_z','image')

	if key == ord("n"):
		move_bounding_box(save=True)
		file_counter = len(sorted(os.listdir("./labels")))
		image = cv2.imread(os.path.join(args["imagesFolder"], files[file_counter]))
		clone = image.copy()
		move_bounding_box()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
 
# close all open windows
cv2.destroyAllWindows()