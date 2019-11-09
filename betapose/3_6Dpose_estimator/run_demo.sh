CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../output01/eval --outdir ../outputEval/ --sp --profile

CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../ape_gen_annotated_01/eval --kpdWeights ape_gen --outdir ../ape_gen_output/ --sp --profile


CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_base_annotated_01/eval --outdir ../kuka_output/ --sp --profile --kpdWeights model_best --sixd_base '/media/daniel/Samsung Flash/datasets_gen/betapose/kuka/kpd'


CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_base_real_annotated_01/eval --outdir ../kuka_real_output/ --sp --profile --kpdWeights model_best --sixd_base ../kuka_real_small





CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_real_full_annotated_01/eval --outdir ../kuka_real_full_output/ --sp --profile --kpdWeights model_best --sixd_base ../kuka_real

CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_real_far_annotated_01/eval --outdir ../kuka_real_far_output/ --sp --profile --kpdWeights model_best --sixd_base ../kuka_real_far

CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_real_near_annotated_01/eval --outdir ../kuka_real_near_output/ --sp --profile --kpdWeights model_best --sixd_base ../kuka_real_near

CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_gen_eval_annotated_01/eval --outdir ../kuka_gen_eval_output/ --sp --profile --kpdWeights model_best --sixd_base ../kuka_gen_eval














CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_real_demo_annotated_01/eval --outdir ../kuka_real_demo_output/ --sp --profile --kpdWeights model_best_trd_135_76 --sixd_base ../kuka_real_demo

CUDA_VISIBLE_DEVICES=0 python betapose_evaluate.py --nThreads 1 --nClasses 50 --indir ../kuka_real_demo2_annotated_01/eval --outdir ../kuka_real_demo2_output/ --sp --profile --kpdWeights model_best_trd_135_76 --sixd_base ../kuka_real_demo2




