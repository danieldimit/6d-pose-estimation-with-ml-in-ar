import subprocess, os

weights_dir = '/home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/3_6Dpose_estimator/train_YOLO/backup/psp/'
darknet_dir = '/home/daniel/Documents/Github/darknet'

cfg_file = '/home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/3_6Dpose_estimator/train_YOLO/cfg/yolo-psp-single.cfg'
data_file = '/home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/3_6Dpose_estimator/train_YOLO/data/psp.data'
#os.chdir()
os.chdir(darknet_dir)


#result = subprocess.run(['ls','-l'], stdout=subprocess.PIPE)
os.remove("yoloReport.csv")

with open('./yoloReport.csv', 'w') as csv_file:
	p = None
	rec = None
	f1 = None
	tp = None
	fp = None
	fn = None
	iou = None
	map05 = None
	epoch = None


	csv_file.write('epoch,precision,recall,F1 score,TP,FP,FN,avg IoU,mAP\n')
	counter = 0

	for weight_file in os.listdir(weights_dir):
		print(str(counter) + ' out of ' + str(len(os.listdir(weights_dir))) + ' analyzed')
		counter+=1

		if (weight_file is 'init.weights'):
			epoch = 0
		else:
			epoch = weight_file.split('_')[1].split('.')[0]
		backup_file = weights_dir + weight_file
		print(weights_dir + weight_file)
		result = subprocess.run(['./darknet','detector', 'map', data_file ,cfg_file, backup_file], stdout=subprocess.PIPE).stdout.decode('utf-8')
		for x in result.splitlines():
			if ("F1-score" in x):
				tmp_split = x.replace(",", "").replace(".", ",").split()
				p = tmp_split[6]
				rec = tmp_split[9]
				f1 = tmp_split[-1]

			if ("average IoU" in x):
				tmp_split = x.replace(",", "").replace(".", ",").split()
				tp = tmp_split[6]
				fp = tmp_split[9]
				fn = tmp_split[12]
				iou = tmp_split[-2]

			if ("mean average precision (mAP@" in x):
				tmp_split = x.replace(",", "").replace(".", ",").split()
				map05 = tmp_split[-4]
				csv_file.write('"' + epoch + '","' + p + '","' + rec + '","' 
					+ f1 + '","' + tp + '","' + fp + '","' + fn + '","' + iou + '","' + map05 + '"\n')
				break

#./darknet detector map data/psp.data cfg/yolo-linemod-single.cfg backup/psp/yolo-linemod-single_5700.weights
#darknet.exe detector map data/obj.data yolo-obj.cfg backup\yolo-obj_7000.weights
