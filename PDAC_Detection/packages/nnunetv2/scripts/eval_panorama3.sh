#!/bin/bash

# conda activate nnunet
module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12
export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"


IMAGE_DIR=/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_TVCE_checkpoints__nnUNetPlans_Res__3d_fullres
# EPOCHS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 final best)
EPOCHS=(700 750 800 850 900 950 final best)
OUTPUT_DIR=/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results/Dataset105_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_TVCE_checkpoints_lesion__nnUNetPlans_Res_lesion__3d_fullres
# EPOCHS=(final)
# FOLDS=(0 1 2 3 4)
FOLDS=(0)

for FOLD in "${FOLDS[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
        nnUNetv2_predict -i ${IMAGE_DIR}/fold_${FOLD}/validation_images \
            -o ${OUTPUT_DIR}/fold_${FOLD}/validation_check_${EPOCH} \
            -d 105 \
            -p nnUNetPlans_Res_lesion \
            -tr nnUNetTrainer_Loss_TVCE_checkpoints_lesion \
            -chk checkpoint_${EPOCH}.pth \
            -f ${FOLD} \
            -c 3d_fullres \
            --save_probabilities 
    done
done