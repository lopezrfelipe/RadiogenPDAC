#!/bin/bash


# CODES=("1 1 1 1" "0 1 1 1" "1 0 1 1" "1 1 0 1" "1 1 1 0" "0 0 1 1" "0 1 0 1" "0 1 1 0" "1 0 0 1" "1 0 1 0" "1 1 0 0" "0 0 0 1" "1 0 0 0" "0 1 0 0" "0 0 1 0")
CODES=("1 1 1 1 0" "0 1 1 1 0" "1 0 1 1 0" "1 1 0 1 0" "1 1 1 0 0" "0 0 1 1 0" "0 1 0 1 0" "0 1 1 0 0" "1 0 0 1 0" "1 0 1 0 0" "1 1 0 0 0" "0 0 0 1 0" "1 0 0 0 0" "0 1 0 0 0" "0 0 1 0 0")
DATASET_NAMES=(MSSEG16)
# DATASET_NAMES=(ISBI UMCL)
# TRAINER_POSTS=(MD_500epochs MDP_500epochs DynKD_500epochs DynKDv2_500epochs)
TRAINER_POSTS=(DynKDv3_500epochs DynKDv4_500epochs)


for dataset_name in "${DATASET_NAMES[@]}"
do
	if [ "$dataset_name" = "ISBI" ]; then
		dataset_id=1
	elif [ "$dataset_name" = "MSSEG16" ]; then
		dataset_id=2
	elif [ "$dataset_name" = "UMCL" ]; then
		dataset_id=3
	else
		echo "Invalid dataset name for nnU-Net"
	fi

	input_dir=/data/nnUNetV2/data/nnUNet_raw/Dataset00${dataset_id}_${dataset_name}/imagesTs
	output_dir=/data/nnUNetV2/data/seg_outputs/Dataset00${dataset_id}_${dataset_name}
	gt_dir=/data/nnUNetV2/data/nnUNet_raw/Dataset00${dataset_id}_${dataset_name}/labelsTs	

	for trainer_post in "${TRAINER_POSTS[@]}"
	do
		i=1
		for code in "${CODES[@]}"
		do
			echo -e "\n**************************************************\nTrainer: $trainer_post, Code: $code, (${i}/${#CODES[@]})\n**************************************************"
			echo "predicting segmentation masks..."
			nnUNetv2_predict -i ${input_dir} -o "${output_dir}/${trainer_post}/${code}" -d ${dataset_id} -c 3d_fullres -f 0 -tr nnUNetTrainer$trainer_post -code $code
			echo -e "\nstart calculating evaluation metrics..."
			python /data/MDPP/src/evaluate.py -pred "${output_dir}/${trainer_post}/${code}" -tar $gt_dir
			i=$((i + 1))
		done
	done
done



# nnUNetv2_predict -i /data/nnUNetV2/data/nnUNet_raw/Dataset002_MSSEG16/imagesTs -o /data/nnUNetV2/data/seg_outputs/Dataset002_MSSEG16/MDP_500epochs/10010 -d 2 -c 3d_fullres -f 0 -tr nnUNetTrainerMDP_500epochs -code 1 0 0 1 0
python /data/MDPP/src/evaluate.py -pred /data/nnUNetV2/data/seg_outputs/Dataset002_MSSEG16/MDP_500epochs/10010 -tar /data/nnUNetV2/data/nnUNet_raw/Dataset002_MSSEG16/labelsTs