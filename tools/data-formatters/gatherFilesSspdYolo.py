import glob, os
import shutil
import json
import numpy as np
import random
import cv2
from distutils.dir_util import copy_tree

imageWidth = 640
imageHeight = 480

def extractRange(xArr):
	return np.max(xArr) - np.min(xArr)

def createJPEGImagesAndLabelsJSONFoldersAndContent():
	if not os.path.exists('../sspdFormat'):
		os.makedirs('../sspdFormat')
		os.makedirs('../sspdFormat/JPEGImages')
		os.makedirs('../sspdFormat/maskPolyColor')
		os.makedirs('../sspdFormat/labelsJSON')
		os.makedirs('../sspdFormat/labels')

	allSubdirs = [x[0] for x in os.walk('./')]
	counter = 0
	counterJson = 0
	counterCs = 0
	for dir in allSubdirs:
		for file in os.listdir(dir):
			if file.endswith(".json") and not file.endswith("settings.json"):
				#shutil.copy(os.path.join(dir, file), os.path.join('../sspdFormat/labelsJSON', format(counterJson, '06') + '.json'))
				counterJson += 1
			if file.endswith(".png") and not file.endswith("cs.png") and not file.endswith("depth.png") and not file.endswith("is.png"):
				#shutil.copy(os.path.join(dir, file), os.path.join('../sspdFormat/JPEGImages', format(counter, '06') + '.png'))
				counter += 1
			if file.endswith("cs.png"):
				shutil.copy(os.path.join(dir, file), os.path.join('../sspdFormat/maskPolyColor', format(counter, '06') + '.cs.png'))
				counterCs += 1

def createLabelContent():
	allSubdirs = [x[0] for x in os.walk('../sspdFormat/labelsJSON')]
	counter = 0
	createdCounter = 0
	for dir in allSubdirs:
		for file in os.listdir(dir):
			with open(os.path.join(dir, file)) as json_file:  
				data = json.load(json_file)
				created = False

				for obj in data['objects']:
					c_x = obj['projected_cuboid_centroid'][0] / imageWidth
					c_y = obj['projected_cuboid_centroid'][1] / imageHeight

					if (c_x <= 1 and c_x >= 0 and c_y <= 1 and c_y >= 0):
						bb_x1 = obj['projected_cuboid'][0][0] / imageWidth
						bb_y1 = obj['projected_cuboid'][0][1] / imageHeight

						bb_x2 = obj['projected_cuboid'][1][0] / imageWidth
						bb_y2 = obj['projected_cuboid'][1][1] / imageHeight

						bb_x3 = obj['projected_cuboid'][2][0] / imageWidth
						bb_y3 = obj['projected_cuboid'][2][1] / imageHeight

						bb_x4 = obj['projected_cuboid'][3][0] / imageWidth
						bb_y4 = obj['projected_cuboid'][3][1] / imageHeight

						bb_x5 = obj['projected_cuboid'][4][0] / imageWidth
						bb_y5 = obj['projected_cuboid'][4][1] / imageHeight

						bb_x6 = obj['projected_cuboid'][5][0] / imageWidth
						bb_y6 = obj['projected_cuboid'][5][1] / imageHeight

						bb_x7 = obj['projected_cuboid'][6][0] / imageWidth
						bb_y7 = obj['projected_cuboid'][6][1] / imageHeight

						bb_x8 = obj['projected_cuboid'][7][0] / imageWidth
						bb_y8 = obj['projected_cuboid'][7][1] / imageHeight

						range_x = extractRange(np.array([bb_x1,bb_x2,bb_x3,bb_x4,bb_x5,bb_x6,bb_x7,bb_x8]))
						range_y = extractRange(np.array([bb_y1,bb_y2,bb_y3,bb_y4,bb_y5,bb_y6,bb_y7,bb_y8]))

						f = open(os.path.join('../sspdFormat/labels',format(createdCounter, '06') + '.txt'), "w+")
						f.write("0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f" % (c_x, c_y, bb_x1, bb_y1, bb_x2, bb_y2, bb_x3, bb_y3, bb_x4, bb_y4, bb_x5, bb_y5, bb_x6, bb_y6, bb_x7, bb_y7, bb_x8, bb_y8, range_x, range_y))
						created = True
						createdCounter += 1
						f.close()
						break
				if not created:
					if (os.path.isfile('../sspdFormat/JPEGImages/' + format(counter, '06') + '.png')):
						os.remove('../sspdFormat/JPEGImages/' + format(counter, '06') + '.png')
					if (os.path.isfile('../sspdFormat/mask/' + format(counter, '06') + '.png')):
						os.remove('../sspdFormat/mask/' + format(counter, '06') + '.png')
				
				counter += 1

def renumberInFolder(folder):
	allSubdirs = [x[0] for x in os.walk(folder)]
	counter = 0
	for dir in allSubdirs:
		for file in os.listdir(dir):
			os.rename(folder + file, folder + format(counter, '06') + '.png')
			counter+=1

def createBinaryMask():
	allSubdirs = [x[0] for x in os.walk('../sspdFormat/labelsJSON')]
	counter = 0
	for dir in allSubdirs:
		for file in os.listdir(dir):
			if os.path.isfile("../sspdFormat/mask/" + format(counter, '06') + ".png"):
				counter += 1
				continue
			with open(os.path.join(dir, file)) as json_file:  
				data = json.load(json_file)
				created = False

				for obj in data['objects']:
					c_x = int(obj['projected_cuboid_centroid'][0])
					c_y = int(obj['projected_cuboid_centroid'][1])
					

					if (c_x <= imageWidth and c_x >= 0 and c_y <= imageHeight and c_y >= 0):
						bb_x1 = int(obj['bounding_box']['top_left'][0])
						bb_y1 = int(obj['bounding_box']['top_left'][1])

						bb_x2 = int(obj['bounding_box']['bottom_right'][0])
						bb_y2 = int(obj['bounding_box']['bottom_right'][1])

						img = cv2.imread("../sspdFormat/maskPolyColor/" + format(counter, '06') + ".cs.png")
						if (c_y == imageHeight):
							c_y -= 1
						if (c_x == imageWidth):
							c_x -= 1
						pspColor = np.array(img[c_y, c_x])

						# Everything outside of the bb is black
						img[0:bb_x1,0:imageWidth] = (0,0,0)
						img[bb_x2:imageHeight,0:imageWidth] = (0,0,0)
						img[0:imageHeight,0:bb_y1] = (0,0,0)
						img[0:imageHeight,bb_y2:imageWidth] = (0,0,0)

						# Precision color the psp white and everything else in the bb black
						for i in range(bb_x1, bb_x2):
							for j in range(bb_y1, bb_y2):
								if (i < imageHeight and i >= 0 and j < imageWidth and j >= 0):
									if np.any(img[i, j] == pspColor):
										img[i, j] = (255,255,255)
									else:
										img[i, j] = (0,0,0)

						#cv2.imshow('title',img)
						cv2.imwrite("../sspdFormat/mask/" + format(counter, '06') + ".png", img)
						created = True
						break
				if (not created):
					img = np.zeros((imageHeight,imageWidth,3), np.uint8)
					cv2.imwrite("../sspdFormat/mask/" + format(counter, '06') + ".png", img)
				counter += 1

def createTestAndTrainFiles(counter):
	test_size = int(counter * 0.05)
	step = int(counter / test_size)
	accOffset = 0
	
	f_test = open(os.path.join('../sspdFormat', 'test.txt'), "w+")
	f_train = open(os.path.join('../sspdFormat', 'train.txt'), "w+")
	f_train_range = open(os.path.join('../sspdFormat', 'training_range.txt'), "w+")
	
	for x in range(test_size):
		accOffsetNew = accOffset + step
		test_object_n = random.randint(accOffset,accOffsetNew)
		f_test.write('sspdFormat/JPEGImages/' + format(test_object_n, '06') + ".png \n")
		for train_object_n in range(accOffset, accOffsetNew):
			if (train_object_n != test_object_n):
				f_train.write('sspdFormat/JPEGImages/' + format(train_object_n, '06') + ".png \n")
				f_train_range.write(str(train_object_n) + " \n")
		accOffset = accOffsetNew
	
	f_test.close()
	f_train.close()
	f_train_range.close()

def createTestAndTrainFilesReduced(counter, reducedAmount):
	test_size = int(reducedAmount)
	step = int(counter / test_size)
	accOffset = 0
	
	f_test = open(os.path.join('../sspdFormat', 'test.txt'), "w+")
	f_train = open(os.path.join('../sspdFormat', 'train.txt'), "w+")
	f_train_range = open(os.path.join('../sspdFormat', 'training_range.txt'), "w+")
	
	for x in range(test_size):
		accOffsetNew = accOffset + step
		test_object_n = random.randint(accOffset,accOffsetNew)
		f_test.write('sspdFormat/JPEGImages/' + format(test_object_n, '06') + ".png \n")
		train_object_n = random.randint(accOffset,accOffsetNew)
		f_train.write('sspdFormat/JPEGImages/' + format(train_object_n, '06') + ".png \n")
		f_train_range.write(str(train_object_n) + " \n")
		accOffset = accOffsetNew
	
	f_test.close()
	f_train.close()
	f_train_range.close()

def calc_pts_diameter(pts):
	diameter = -1
	for pt_id in range(pts.shape[0]):
		pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
		pts_diff = pt_dup - pts[pt_id:, :]
		max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
		if max_dist > diameter:
			diameter = max_dist
	return diameter

def copySSPD():
	fromDirectory = "../sspdFormat"
	toDirectory = "../betaposeFormat"

	copy_tree(fromDirectory, toDirectory)

def changeLabels():
	for file in glob.glob("../betaposeFormat/labels/*.txt"):
	    f = open(file, "r")
	    line = f.read()
	    lineVals = line.split()
	    newLine = lineVals[0] + ' ' + lineVals[1] + ' ' + lineVals[2] + ' ' + lineVals[19] + ' ' + lineVals[20]
	    with open('../betaposeFormat/labelsConverted/' + os.path.basename(f.name), 'w') as file:
	        file.write(newLine)

def cleanUselessFoldersBetapose():
	shutil.rmtree('../betaposeFormat/maskPolyColor/')
	shutil.rmtree('../betaposeFormat/labels/')
	shutil.rmtree('../betaposeFormat/labelsJSON/')

def cleanUselessFoldersSSPD():
	shutil.rmtree('../betaposeFormat/maskPolyColor/')
	shutil.rmtree('../betaposeFormat/labelsJSON/')








def reformatForSSPD():
	createJPEGImagesAndLabelsJSONFoldersAndContent()
	createBinaryMask()
	createLabelContent()
	renumberInFolder('../sspdFormat/mask/')
	renumberInFolder('../sspdFormat/JPEGImages/')
	createTestAndTrainFiles(len(os.listdir('../sspdFormat/labels')))
	#createTestAndTrainFilesReduced(len(os.listdir('../sspdFormat/labels')), 2000)
	cleanUselessFoldersSSPD()

def reformatForYoloAndBetapose():
	reformatForSSPD()
	copySSPD()
	changeLabels()
	cleanUselessFoldersBetapose()
	os.rename('../betaposeFormat/labelsConverted/', '../betaposeFormat/labels/')

createTestAndTrainFilesReduced(len(os.listdir('../sspdFormat/labels')), 750)