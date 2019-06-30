CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID psp_noDPG --optMethod adam --nEpochs 2000
CUDA_VISIBLE_DEVICES=0  python src/train.py --trainBatch 4 --expID psp_DPG --optMethod adam --nEpochs 2000 --loadModel ../exp/coco/new_seq1_50kp_2/model_200.pkl --addDPG


CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID gibson_noDPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/gibson_annotated_01
