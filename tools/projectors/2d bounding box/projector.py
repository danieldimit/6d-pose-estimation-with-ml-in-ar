boundingBox = [304.03497314453125, 283.9568176269531, 369.9814453125, 348.4409484863281]
import numpy as np
import cv2

img = cv2.imread('test.png', 1)
cv2.rectangle(img,(304, 283),(369, 348),(0,255,0),3)
cv2.imwrite('result.png',img)
