import os
import SimpleITK as sitk
import numpy as np
import pickle


fp = "/pct_wbo2/home/han.l/Panorama/nnDetection/models/Task001D3_PDAC/RetinaUNetV001_D3V001_3d/consolidated/test_predictions/100054_00001_boxes.pkl"


with open(fp, 'rb') as f:
    data = pickle.load(f)
    labels = data["pred_labels"]
    scores = data["pred_scores"]
    boxes = data["pred_boxes"]

    scores = scores[labels==0]
    boxes = boxes[labels==0]
    breakpoint()