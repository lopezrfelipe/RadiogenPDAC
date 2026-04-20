#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --time=5-0:00:00 
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --output=nnUNetMD_DV8.stdout
#SBATCH --job-name=nnUNetMD_DV8


conda activate nnunetv2


export nnUNet_raw="YourPath/nnUNet_raw" 
export nnUNet_preprocessed="YourPath/nnUNet_preprocessed" 
export nnUNet_results="YourPath/nnUNet_results"


A=(nnUNetTrainerKD_DV8_v1 nnUNetTrainerKD_DV8_v2 nnUNetTrainerKD_DV8_v3 nnUNetTrainerKD_DV8_v4 nnUNetTrainerKD_DV8_v5 nnUNetTrainerKD_DV8_v6 nnUNetTrainerKD_DV8_v1_co nnUNetTrainerKD_DV8_v2_co nnUNetTrainerKD_DV8_v3_co nnUNetTrainerKD_DV8_v4_co nnUNetTrainerKD_DV8_v5_co nnUNetTrainerKD_DV8_v6_co)


nnUNetv2_train 4 3d_fullres 0 -tr ${A[$SLURM_ARRAY_TASK_ID]} 