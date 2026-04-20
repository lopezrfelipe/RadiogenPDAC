#!/bin/bash
#SBATCH --account=rctcu02219 
#SBATCH --partition=defq-large
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=6
#SBATCH --mem=96G
#SBATCH --job-name=flare2


module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12
source activate nnunet

export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"

hostname; date

nnUNetv2_train 600 3d_fullres 4

echo "done"
