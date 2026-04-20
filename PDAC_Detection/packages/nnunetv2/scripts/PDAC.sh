#!/bin/bash
#SBATCH --account=rctcu02219 
#SBATCH --partition=a100-4gpus-small
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=6
#SBATCH --mem=96G
#SBATCH --job-name=pdac


module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12
source activate nnunet

export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"

hostname; date

CUDA_VISIBLE_DEVICES=0 
nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainer_Loss_CE_checkpoints --c --npz &

CUDA_VISIBLE_DEVICES=1
nnUNetv2_train 104 3d_fullres 1 -tr nnUNetTrainer_Loss_CE_checkpoints --c --npz &

CUDA_VISIBLE_DEVICES=2
nnUNetv2_train 104 3d_fullres 2 -tr nnUNetTrainer_Loss_CE_checkpoints --c --npz &

CUDA_VISIBLE_DEVICES=3
nnUNetv2_train 104 3d_fullres 3 -tr nnUNetTrainer_Loss_CE_checkpoints --c --npz &

wait 
echo "done"
