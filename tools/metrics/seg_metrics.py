from datetime import timedelta
from glob import glob
from multiprocessing import Pool
from os.path import basename, join
from time import perf_counter

import numpy as np
import pandas as pd
import SimpleITK as sitk
from cc3d import connected_components
from natsort import natsorted


def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = sitk.ReadImage(nii_path)
    mask = sitk.GetArrayFromImage(mask_nii)
    voxel_vol = np.prod(mask_nii.GetSpacing()) / 1000
    return mask, voxel_vol


def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18
    conn_comp = connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def false_pos_pix(gt_array, pred_array):
    # compute number of voxels of false positive connected components in prediction mask
    pred_conn_comp = con_comp(pred_array)

    false_pos = 0
    for idx in range(1, pred_conn_comp.max() + 1):
        comp_mask = np.isin(pred_conn_comp, idx)
        if np.sum(comp_mask * gt_array) == 0:
            false_pos += np.sum(comp_mask)
    return false_pos


def false_neg_pix(gt_array, pred_array):
    # compute number of voxels of false negative connected components (of the ground truth mask) in the prediction mask
    gt_conn_comp = con_comp(gt_array)

    false_neg = 0
    for idx in range(1, gt_conn_comp.max() + 1):
        comp_mask = np.isin(gt_conn_comp, idx)
        if np.sum(comp_mask * pred_array) == 0:
            false_neg += np.sum(comp_mask)

    return false_neg


def dice_score(mask1, mask2):
    # compute foreground Dice coefficient
    overlap = (mask1 * mask2).sum()
    sum = mask1.sum() + mask2.sum()
    dice_score = 2 * overlap / sum
    return dice_score


def compute_metrics(sample_id, nii_gt_path, nii_pred_path):
    # main function
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    # Threshold to binary [0, 1]
    gt_array = (gt_array != 0).astype(np.uint8)
    pred_array = (pred_array != 0).astype(np.uint8)

    false_neg_vol = false_neg_pix(gt_array, pred_array) * voxel_vol
    false_pos_vol = false_pos_pix(gt_array, pred_array) * voxel_vol

    mtv_gt = np.count_nonzero(gt_array) * voxel_vol
    mtv_pred = np.count_nonzero(pred_array) * voxel_vol

    if np.sum(gt_array) == 0:
        dice_sc = np.nan
    else:
        dice_sc = dice_score(gt_array, pred_array)

    df = pd.DataFrame(
        {
            "SampleID": [sample_id],
            "GT_Path": [nii_gt_path],
            "PRED_Path": [nii_pred_path],
            "Dice": [dice_sc],
            "FPV": [false_pos_vol],
            "FNV": [false_neg_vol],
            "MTV_GT": [mtv_gt],
            "MTV_PRED": [mtv_pred],
        }
    )
    print(f"SampleID: {sample_id}")
    print(f"GT_Path: {nii_gt_path}")
    print(f"PRED_Path: {nii_pred_path}")
    print(f"Dice: {dice_sc:.2f}")
    print(f"FPV: {false_pos_vol:.1f}")
    print(f"FNV: {false_neg_vol:.1f}")
    print(f"MTV GT: {mtv_gt:.1f}")
    print(f"MTV PRED: {mtv_pred:.1f}")
    print()
    return df


if __name__ == "__main__":
    """
    From here: https://github.com/lab-midas/autoPET/blob/master/val_script.py
    Adapted for multiprocessing
    """
    nii_gt_dir = "/path/to/gt"
    nii_pred_dir = "/path/to/pred"
    save_as = "/path/to/out/seg_metrics.csv"
    n_jobs = 8
    print(f"Executing on {n_jobs} cores.")

    nii_pred_paths = natsorted(glob(join(nii_pred_dir, "*.nii*")))
    nii_gt_paths = natsorted(glob(join(nii_gt_dir, "*.nii*")))

    mp_args = []
    for nii_gt_path, nii_pred_path in zip(nii_gt_paths, nii_pred_paths):
        mp_args.append((basename(nii_gt_path).split(".")[0], nii_gt_path, nii_pred_path))

    # Start time
    start_time = perf_counter()

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        mp_df = p.starmap(compute_metrics, mp_args)

    # End time
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    elapsed_time = timedelta(seconds=elapsed_time)
    print(f"Done. Took {elapsed_time.seconds} seconds or {elapsed_time.seconds/60:.0f} minutes.")

    out_df = pd.concat(mp_df)
    out_df.to_csv(save_as, index=False)
