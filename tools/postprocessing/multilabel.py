from glob import glob
from multiprocessing import Pool
from os import makedirs
from os.path import basename, dirname, isfile, join

import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted


def read_nifti(filename):
    sitk_img = sitk.ReadImage(filename)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    img_arr = np.transpose(img_arr)
    return sitk_img, img_arr


def combine_masks(masks, save_as):
    print(f"Processing: {save_as}")
    if isfile(save_as):
        pass
    else:
        # Read one segmentation as reference
        sitk_mask, mask_arr = read_nifti(masks[0])

        # Merge
        out_mask = np.zeros_like(mask_arr)
        for mask in masks:
            sitk_mask, mask_arr = read_nifti(mask)
            out_mask = sum([out_mask, mask_arr])

        # Threshold to 1
        out_mask = np.where(out_mask != 0, 1, 0).astype(np.uint32)

        # To sitk object
        out_mask = np.transpose(out_mask)
        sitk_out = sitk.GetImageFromArray(out_mask)
        sitk_out.CopyInformation(sitk_mask)

        # Save as nifti
        output_dir = dirname(masks[0])
        sitk.WriteImage(sitk_out, join(output_dir, save_as))


def combine_from_list(csv_path, input_dir, output_dir, n_jobs=1):
    """
    Combines multiple segmentation masks as defined in the input csv table.

    CSV-File:
    col1 named "FILENAME" (filename of mask, e.g.: rib_left_1.nii.gz)
    col2 named "GROUP" (e.g., ribs)

    Parameters
    ----------
    csv_path : str
        Path to input csv file
    input_dir : str
        Path to input directory; contains folders with TS-output.
    output_dir : str
        Path to output directory, your results will be saved here
    n_jobs : int, optional
        Number of CPU cores to use, by default 1. Dont go crazy, otherwise you run OOM.
    """

    print(f"Executing on {n_jobs} cores.")

    # Read csv file
    df = pd.read_csv(csv_path)

    folder_paths = natsorted(glob(join(input_dir, "*")))

    mp_args = []

    for folder_path in folder_paths:
        folder = basename(folder_path)

        # Make output folder
        makedirs(join(output_dir, folder), exist_ok=True)

        for group in df["GROUP"].unique():
            temp_df = df[df["GROUP"] == group]

            masks = []
            for label_name in temp_df["FILENAME"]:
                masks.append(join(folder_path, label_name))

            save_as = join(output_dir, folder, f"{group}.nii.gz")

            mp_args.append((masks, save_as))

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        p.starmap(combine_masks, mp_args)


if __name__ == "__main__":

    combine_from_list(
        csv_path="/path/to/csv/input_multilabel.csv",
        input_dir="/path/to/inp_dir",
        output_dir="/path/to/out_dir",
        n_jobs=24,
    )
