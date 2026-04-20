#!/bin/bash
#SBATCH --account=rctcu02344 
#SBATCH --partition=a100-4gpu-40gb
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=6
#SBATCH --mem=96G
#SBATCH --job-name=pano4
#SBATCH --output=pano4.out
#SBATCH --exclude=node189


module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12
source activate nnunet
export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"

hostname; date

CUDA_VISIBLE_DEVICES=0
nnUNetv2_train 107 3d_fullres 2 -tr nnUNetTrainer_WFocalLoss -p nnUNetPlans_v2 &

CUDA_VISIBLE_DEVICES=1
nnUNetv2_train 107 3d_fullres 3 -tr nnUNetTrainer_WFocalLoss -p nnUNetPlans_v2 &

CUDA_VISIBLE_DEVICES=2
nnUNetv2_train 107 3d_fullres 4 -tr nnUNetTrainer_WFocalLoss -p nnUNetPlans_v2 &

wait 
echo "done"
