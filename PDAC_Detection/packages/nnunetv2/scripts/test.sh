#!/bin/bash
export nnUNet_raw="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_raw" 
export nnUNet_preprocessed="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_preprocessed" 
export nnUNet_results="/pct_wbo2/home/han.l/nnUNetMD/workspace/nnUNet_results"


nnUNetv2_predict -i /pct_wbo2/home/han.l/FLARE/PublicValidation/imagesVal -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold0 -d 600 -c 3d_fullres -f 0 --save_probabilities
nnUNetv2_predict -i /pct_wbo2/home/han.l/FLARE/PublicValidation/imagesVal -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold1 -d 600 -c 3d_fullres -f 1 --save_probabilities
nnUNetv2_predict -i /pct_wbo2/home/han.l/FLARE/PublicValidation/imagesVal -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold2 -d 600 -c 3d_fullres -f 2 --save_probabilities
nnUNetv2_predict -i /pct_wbo2/home/han.l/FLARE/PublicValidation/imagesVal -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold3 -d 600 -c 3d_fullres -f 3 --save_probabilities
nnUNetv2_predict -i /pct_wbo2/home/han.l/FLARE/PublicValidation/imagesVal -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold4 -d 600 -c 3d_fullres -f 4 --save_probabilities
nnUNetv2_ensemble -i /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold0 /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold1 /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold2 /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold3 /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold4 -o /pct_wbo2/home/han.l/FLARE/PublicValidation/submit_ensemble 