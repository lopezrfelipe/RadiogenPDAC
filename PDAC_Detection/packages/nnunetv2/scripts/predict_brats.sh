#!/bin/bash


CODES=("1 1 1 1" "0 1 1 1" "1 0 1 1" "1 1 0 1" "1 1 1 0" "0 0 1 1" "0 1 0 1" "0 1 1 0" "1 0 0 1" "1 0 1 0" "1 1 0 0" "0 0 0 1" "1 0 0 0" "0 1 0 0" "0 0 1 0")
DATASET_NAME=(BraTS2023)
TRAINER_POSTS=(KD_DV8_V6 KD_DV7_V3 KD_DV7_V4 KD_DV7_V5 KD_DV7_V6) # KD_DV8_V1 KD_DV8_V3 KD_DV8_V4 KD_DV7_V3 KD_DV7_V4

input_dir=/data/nnUNetV2/data/nnUNet_raw/Dataset004_BraTS2023/imagesTs
output_dir=/data/nnUNetV2/data/seg_outputs/Dataset004_BraTS2023
gt_dir=/data/nnUNetV2/data/nnUNet_raw/Dataset004_BraTS2023/labelsTs	

# python -W ignore /data/MDPP/src/evaluate_brats.py -pred /data/nnUNetV2/data/seg_outputs/Dataset004_BraTS2023/nnUNetTrainer/1111 -tar /data/nnUNetV2/data/nnUNet_raw/Dataset004_BraTS2023/labelsTs

for trainer_post in "${TRAINER_POSTS[@]}"
do
	i=1
	for code in "${CODES[@]}"
	do
		echo -e "\n**************************************************\nTrainer: $trainer_post, Code: $code, (${i}/${#CODES[@]})\n**************************************************"
		echo "predicting segmentation masks..."
		nnUNetv2_predict -i ${input_dir} -o "${output_dir}/${trainer_post}/${code}" -d 4 -c 3d_fullres -f 0 -tr nnUNetTrainer$trainer_post -code $code --disable_tta
		echo -e "\nstart calculating evaluation metrics..."
		python -W ignore /data/MDPP/src/evaluate_brats.py -pred "${output_dir}/${trainer_post}/${code}" -tar $gt_dir
		i=$((i + 1))
	done
done

# nnUNetv2_predict -i /data/nnUNetV2/data/nnUNet_raw/Dataset004_BraTS2023/imagesTs -o /data/nnUNetV2/data/seg_outputs/Dataset004_BraTS2023/nnUNetTrainer/1111 -d 4 -c 3d_fullres -f 0 -tr nnUNetTrainer

