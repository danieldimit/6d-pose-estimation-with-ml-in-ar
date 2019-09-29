import glob, os, sys
import json
import numpy as np

def createLabelContent():
	print('creating labels and deleting pics where the obj is partly outside of frame')
	f = open('gt.yml', "w+")
	f_c = open('info.yml', "w+")

	with open('annotation-results.6dan.json.json', 'r') as json_file:
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


def createTestAndTrainFiles(counter):
	print('creating test files')

	f_test = open(os.path.join('./', 'test.txt'), "w+")
	for i in range(counter):
		img_type = ".jpg"
		f_test.write('./JPEGImages/' + format(i, '06') + img_type + " \n")
	
	f_test.close()








	
if __name__ == "__main__":
	createLabelContent()
	createTestAndTrainFiles(len(os.listdir('./manual')))

