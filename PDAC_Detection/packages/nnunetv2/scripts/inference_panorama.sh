#!/bin/bash

module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12
# source activate nnunet

export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"

RESULT_DIR=/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results/Dataset104_PANORAMA_baseline_PDAC_Detection/nnUNetTrainer_Loss_CE_checkpoints__nnUNetPlans__3d_fullres
FOLDS=(0 1 2 3 4)


for FOLD in "${FOLDS[@]}"
do
    EPOCHS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 final)
    for EPOCH in "${EPOCHS[@]}"; do
        nnUNetv2_predict -i ${RESULT_DIR}/fold_${FOLD}/validation_images \
            -o ${RESULT_DIR}/fold_${FOLD}/validation_check_${EPOCH} \
            -d 104 \
            -tr nnUNetTrainer_Loss_CE_checkpoints \
            -chk checkpoint_${EPOCH}.pth \
            -f ${FOLD} \
            -c 3d_fullres \
            --save_probabilities 
    done
done