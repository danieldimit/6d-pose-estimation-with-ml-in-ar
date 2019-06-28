import numpy as np
import cv2

boundingBox = [309, 304, 48, 53]

img = cv2.imread('505.png', 1)
cv2.rectangle(img,(boundingBox[0], boundingBox[1]),(boundingBox[0] + boundingBox[2], boundingBox[1] + boundingBox[3]),(0,255,0),3)
cv2.imwrite('result.png',img)
