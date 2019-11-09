CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID psp_noDPG --optMethod adam --nEpochs 2000
CUDA_VISIBLE_DEVICES=0  python src/train.py --trainBatch 4 --expID psp_DPG --optMethod adam --nEpochs 2000 --loadModel ../exp/coco/new_seq1_50kp_2/model_200.pkl --addDPG



CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID psp_noDPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/ape_gen_annotated_01

CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID psp_DPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/psp_base_annotated_01 --loadModel ../exp/coco/psp_noDPG/model_best.pkl --addDPG








CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_noDPG --optMethod adam --nEpochs 2000 --nKps 49 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_base_annotated_01

CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_DPG --optMethod adam --nEpochs 2000 --nKps 49 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_base_annotated_01 --loadModel ../exp/coco/kuka_noDPG/model_best.pkl --addDPG









CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_noDPG_degrees --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_base_annotated_01

CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_DPG_degrees --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_base_annotated_01 --loadModel ../exp/coco/kuka_DPG_degrees/model_best.pkl --addDPG







CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_real_train_noDPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_real_train_annotated_01 --loadModel ../exp/coco/kuka_DPG_degrees/model_best.pkl

CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_real_train_DPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_real_full_annotated_01 --loadModel ../exp/coco/kuka_real_train_noDPG/model_best.pkl --addDPG

CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 4 --expID kuka_real_train_DPG --optMethod adam --nEpochs 2000 --nKps 50 --inputDir /home/daniel/Documents/Github/methods_6d_pose_estimation/betapose/kuka_real_full_annotated_01 --loadModel ../exp/coco/kuka_real_train_noDPG/model_best_105_70.pkl --addDPG

