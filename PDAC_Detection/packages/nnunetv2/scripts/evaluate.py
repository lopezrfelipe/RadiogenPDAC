import os
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred, gt, label):
    pred_label = (pred == label).astype(int)
    gt_label = (gt == label).astype(int)
    intersection = np.sum(pred_label * gt_label)
    return (2. * intersection) / (np.sum(pred_label) + np.sum(gt_label))

def normalized_surface_dice(pred, gt, label, tolerance_mm=1.0, voxel_spacing=None):
    pred_label = (pred == label).astype(int)
    gt_label = (gt == label).astype(int)

    # Calculate distance transforms
    pred_dist = distance_transform_edt(1 - pred_label, sampling=voxel_spacing)
    gt_dist = distance_transform_edt(1 - gt_label, sampling=voxel_spacing)
    
    pred_surface = np.logical_and(pred_dist == 1, pred_label)
    gt_surface = np.logical_and(gt_dist == 1, gt_label)

    # Tolerance mask
    pred_within_tol = pred_surface * (gt_dist <= tolerance_mm)
    gt_within_tol = gt_surface * (pred_dist <= tolerance_mm)

    pred_nsd = np.sum(pred_within_tol) / (np.sum(pred_surface) + 1e-8)
    gt_nsd = np.sum(gt_within_tol) / (np.sum(gt_surface) + 1e-8)

    return 0.5 * (pred_nsd + gt_nsd)

def compute_metrics(pred_dir, gt_dir, labels, tolerance_mm=1.0):
    dice_scores = {label: [] for label in labels}
    nsd_scores = {label: [] for label in labels}

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)
        
        pred_data = pred_img.get_fdata()
        gt_data = gt_img.get_fdata()

        voxel_spacing = gt_img.header.get_zooms()
        avg = []
        for label in labels:
            dice = dice_score(pred_data, gt_data, label)
            nsd = normalized_surface_dice(pred_data, gt_data, label, tolerance_mm, voxel_spacing)
            
            dice_scores[label].append(dice)
            nsd_scores[label].append(nsd)
            avg.append(dice)
            print(dice)
        print('avg dice: ', np.mean(avg))
        print('-------')
        
    
    avg_dice_scores = {label: np.mean(scores) for label, scores in dice_scores.items()}
    avg_nsd_scores = {label: np.mean(scores) for label, scores in nsd_scores.items()}

    return avg_dice_scores, avg_nsd_scores


if __name__ == "__main__":
    # Example usage
    pred_dir = '/pct_wbo2/home/han.l/FLARE/PublicValidation/submit_fold0'
    gt_dir = '/pct_wbo2/home/han.l/FLARE/PublicValidation/labelsVal'

    labels = list(range(1, 14))  # Labels from 1 to 13
    tolerance_mm = 1.0

    avg_dice_scores, avg_nsd_scores = compute_metrics(pred_dir, gt_dir, labels, tolerance_mm)

    print("Average Dice Scores per Organ:")
    for label, score in avg_dice_scores.items():
        print(f"Label {label}: {score:.4f}")

    print("\nAverage Normalized Surface Dice per Organ:")
    for label, score in avg_nsd_scores.items():
        print(f"Label {label}: {score:.4f}")
