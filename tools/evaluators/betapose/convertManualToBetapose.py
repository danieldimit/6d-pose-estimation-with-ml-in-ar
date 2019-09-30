import glob, os, sys
import json
import numpy as np
import shutil

def createLabelContent():
	print('creating labels and deleting pics where the obj is partly outside of frame')
	f = open('gt.yml', "w+")
	f_c = open('info.yml', "w+")

	with open('annotation-results.6dan.json', 'r') as json_file:
		data = json.load(json_file)
		for lbl in data:
			p = lbl['sspd']
			print(int(lbl['image'].split('/')[-1].split('.')[0]))
			f.write(str(int(lbl['image'].split('/')[-1].split('.')[0])) + ':\n')
			print(('- cam_R_m2c: ' + str(lbl['betapose']['cam_R_m2c']) + '\n'))
			f.write('- cam_R_m2c: ' + str(lbl['betapose']['cam_R_m2c']) + '\n')
			f.write('  cam_t_m2c: ' + np.array2string(np.array(lbl['betapose']['cam_t_m2c'],dtype='float') * 1000, precision=8, separator=',', suppress_small=True) + '\n')
			f.write('  obj_bb: ' + str(lbl['betapose']['obj_bb']) + '\n')
			f.write('  obj_id: 1\n')
			if 'cam_K' in lbl['betapose']:
				f_c.write(str(int(lbl['image'].split('/')[-1].split('.')[0])) + ':\n')
				f_c.write('  cam_K: ' + str(lbl['betapose']['cam_K']) + '\n')
				f_c.write('  depth_scale: 1.0\n')
			else:
				f_c.write(str(int(lbl['image'].split('/')[-1].split('.')[0])) + ':\n')
				f_c.write('  cam_K: [320,0,320,0,320,240,0,0,1]\n')
				f_c.write('  depth_scale: 1.0\n')

	f.close()
	f_c.close()

def addAugmentedData():
	print('creating labels and deleting pics where the obj is partly outside of frame')
	f = open('gt.yml', "a+")
	f_c = open('info.yml', "a+")

	counter = 178
	while (counter < 1958):
		f.write(str(counter) + ':\n')
		f.write('- cam_R_m2c: [-0.08728959178229491, 0.9938541911776673, -0.06807623553829344, 0.7203072468360182, 0.015762964390439546, -0.6934760263405948, -0.6881409719832091, -0.10956904504562694, -0.7172563189305435]\n')
		f.write('  cam_t_m2c: [   0.        , -60.        ,1437.96625949]\n')
		f.write('  obj_bb: [153, 69, 310, 281]\n')
		f.write('  obj_id: 1\n')
		f_c.write(str(counter) + ':\n')
		f_c.write('  cam_K: [614.15252686, 0, 323.15930176, 0, 614.09857178, 239.60848999, 0, 0, 1]\n')
		f_c.write('  depth_scale: 1.0\n')
		shutil.copy('./JPEGImages/' + format(counter, '06') + '.jpg', './kukaFinalBeta/' + format(counter, '04') + '.jpg')
		counter+=1

	f.close()
	f_c.close()






	
if __name__ == "__main__":
	addAugmentedData()

