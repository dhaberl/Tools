from glob import glob
from multiprocessing import Pool
from os.path import basename, join

import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted
from qib import FeatureExtractor


def mp_extract(img_path, seg_path, sample_id):
    out_dict = {}
    roi = FeatureExtractor(img_path, seg_path)
    mtv = roi.mtv()
    out_dict["ID"] = [sample_id]
    out_dict["MTV"] = [mtv]

    return pd.DataFrame(out_dict)


def extract_mtv(img_dir, seg_dir, save_as, n_jobs=1):
    print(f"Executing on {n_jobs} cores.")

    img_paths = natsorted(glob(join(img_dir, "*.nii.gz")))
    seg_paths = natsorted(glob(join(seg_dir, "*.nii.gz")))

    mp_args = []
    for img_path, seg_path in zip(img_paths, seg_paths):
        mp_args.append((img_path, seg_path, basename(seg_path)))

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        out_df = p.starmap(mp_extract, mp_args)

    # Concatenate results
    out_df = pd.concat(out_df, ignore_index=True)
    print(out_df)

    # Save
    out_df.to_csv(save_as, index=False)


def mtv(seg_path):
    sitk_seg = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(sitk_seg)

    # Threshold to binary
    seg_arr = np.where(seg_arr != 0, 1, 0).astype(np.uint32)

    # Count number of voxels which are non-zero
    num_nonzero_voxels = np.count_nonzero(seg_arr)
    # print(f"Number of nonzero voxels: {num_nonzero_voxels}")

    # Calculate volume of a single voxel in mm3
    voxel_vol = np.product(sitk_seg.GetSpacing())
    # print(f"Volume of a single voxel (mm3): {voxel_vol}")

    # Calculate the volume of all non-zero voxels (=metabolic tumor volume)
    nonzero_voxel_vol = num_nonzero_voxels * voxel_vol

    # Convert to cm3 = ml
    nonzero_voxel_vol = nonzero_voxel_vol / 1000
    # print(f"Nonzero voxel volume (cm3 = ml): {nonzero_voxel_vol}")

    return nonzero_voxel_vol


def calc_mtv_from_directory(seg_dir, save_as):
    seg_paths = natsorted(glob(join(seg_dir, "*.nii.*")))

    out_dict = {"ID": [], "MTV_in_ml": []}
    for seg_path in seg_paths:
        print(f"Processing case: {basename(seg_path)}")

        metabolic_tumor_volume = mtv(seg_path)

        out_dict["ID"].append(basename(seg_path).split(".")[0])
        out_dict["MTV_in_ml"].append(metabolic_tumor_volume)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(save_as, index=False)


if __name__ == "__main__":
    # Multiprocessed version
    # extract_mtv(
    #     img_dir="/path/to/img_dir/",
    #     seg_dir="/path/to/seg_dir/",
    #     save_as="/path/to/out_dir/mtv.csv",
    #     n_jobs=24,
    # )

    # From directory, not parallel
    calc_mtv_from_directory(
        seg_dir="path/to/seg_dir",
        save_as="mtv.csv",
    )
