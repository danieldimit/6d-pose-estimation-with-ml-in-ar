# This script takes a label file from LINEMOD and displays
# the training points over the corresponding image
# This is just to understand how the label file should be created.


import cv2
import numpy as np

import os


label_dir = "LINEMOD/ape/labels/000000.txt" #"sspdFormat/labels/000011.txt"
image_dir ="LINEMOD/ape/JPEGImages/000000.jpg"# "sspdFormat/JPEGImages/000011.png" 


def main():
	with open(label_dir) as lb_f:
		label = lb_f.readline()
	#print (label.split())
	label = label.split()

	img = cv2.imread(image_dir)
	print(img.shape)

	# create a window to display image
	wname = "Prediction"
	cv2.namedWindow(wname)	

	# draw bounding cube
	for j,i in enumerate(np.arange(1,19,2)):
		col1 = 28*j
		col2 = 255 - (28*j)
		col3 = np.random.randint(0,256)
		x = int(float(label[i])*img.shape[1])
		y = int(float(label[i+1])*img.shape[0])
		cv2.circle(img,(x,y), 3, (col1,col2,col3), -1)
		cv2.putText(img, str(j+1), (x + 5, y + 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (col1, col2, col3), 1)

	# draw x and y range
	x_r = int(float(label[19])*img.shape[1])
	y_r = int(float(label[20])*img.shape[0])

	cv2.line(img,(int(float(label[17])*img.shape[1]),int(float(label[18])*img.shape[0])),(int(float(label[17])*img.shape[1])+x_r,int(float(label[18])*img.shape[0])+y_r), (0,255,0),1)

	# Show the image and wait key press
	cv2.imshow(wname, img)
	cv2.waitKey()




if __name__ == '__main__':
	main()