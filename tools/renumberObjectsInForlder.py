import glob, os, sys
import shutil
import json
import random
from distutils.dir_util import copy_tree

imageWidth = 640
imageHeight = 480


def renumberInFolder(folder):
	allSubdirs = sorted([x[0] for x in os.walk(folder)])
	print(len(allSubdirs))
	counter = 0
	for dir in allSubdirs:
		for file in sorted(os.listdir(dir)):
			end = file.split('.')[-1]
			os.rename(folder + file, folder + format(counter, '06') + '.' + end)
			counter+=1

if __name__ == "__main__":
    # Training settings
    # example: python bbCalcForLabels.py guitar 1499 gibson10x.ply
    
	renumberInFolder('./training_manual/')
