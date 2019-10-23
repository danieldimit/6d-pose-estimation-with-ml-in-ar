import numpy as np
import cv2, sys

def readAndProjectYoloLabel(name):
	img = cv2.imread(name + '.jpg', 1)
	height, width, channels = img.shape

	with open(name + '.txt') as f:
		content = f.readlines()
		content = [x.strip() for x in content]

		for c in content:
			obj = c.split()[1:]
			obj = [float(i) for i in obj]
			x1 = int(obj[0] * width)
			y1 = int(obj[1] * height)
			x2 = int((obj[2] * width) / 2)
			y2 = int((obj[3] * height) / 2)
			cv2.rectangle(img,(x1 - x2, y1 - y2),(x1 + x2, y1 + y2),(0,255,0),3)

	cv2.namedWindow(name)
	# Show the image and wait key press
	cv2.imshow(name, img)
	cv2.waitKey()

if __name__ == "__main__":
    # Training settings
	# example: python bbCalcForLabels.py guitar 1499 gibson10x.ply
	name     = sys.argv[1]
	readAndProjectYoloLabel(name)
