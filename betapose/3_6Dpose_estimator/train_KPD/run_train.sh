CUDA_VISIBLE_DEVICES=0 python src/train.py --trainBatch 3 --expID new_seq1_50kp_2 --optMethod adam --nEpochs 2000
CUDA_VISIBLE_DEVICES=0  python src/train.py --trainBatch 4 --expID seq1_dpg_April --optMethod adam --nEpochs 2000 --loadModel ../exp/coco/new_seq1_50kp_2/model_495.pkl --addDPG
