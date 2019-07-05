import glob, os, shutil, sys

def createLogs(in_f, out_f):
	print('creating logs')
	if (os.path.isfile(out_f)):
		os.remove(out_f)

	with open(out_f, 'w') as file:
		f = open(in_f, "r")
		me = None
		acc5px = None
		acc3d = None
		acc5cm = None
		te = None
		ae = None
		prevLine = None
		epoch = None

		file.write('epoch,mean error,acc 5px,acc 3d,acc 5cm,translation error,angle error\n')

		for x in f:
			if ("Testing..." in x):
				epoch = prevLine.split()[-1].replace(".", ",")

			if ("Mean corner error" in x):
				me = x.split()[-1].replace(".", ",")

			if ("Acc using 5 px 2D Projection" in x):
				acc5px = x.split()[-1][:-1].replace(".", ",")

			if ("vx 3D Transformation" in x):
				acc3d = x.split()[-1][:-1].replace(".", ",")

			if ("Acc using 5 cm 5 degree metric" in x):
				acc5cm = x.split()[-1][:-1].replace(".", ",")

			if ("Translation error" in x):
				tmp = x.split()
				ae = tmp[-1].replace(".", ",")
				te = tmp[-4][:-1].replace(".", ",")

			if ("save training stats to" in x):
				file.write('"' + epoch + '","' + me + '","' + acc5px + '","' 
					+ acc3d + '","' + acc5cm + '","' + te + '","' + ae + '"\n')
			prevLine = x
		f.close()


if __name__ == "__main__":
    # Training settings
    # example: python bbCalcForLabels.py guitar 1499 gibson10x.ply
    in_f   = sys.argv[1]
    out_f   = sys.argv[2]
    createLogs(in_f, out_f)