import os
from glob import glob
from os.path import basename, join

import numpy as np
import SimpleITK as sitk
from natsort import natsorted


def main():
    """Main func"""

    ct_dir = "path/to/ct_dir"
    ts_dir = "path/to/totalsegmentator_dir"

    ct_paths = natsorted(glob(join(ct_dir, "*.nii*")))

    for ind, ct_path in enumerate(ct_paths):
        print(f"{ind+1}/{len(ct_paths)} {ct_path}")
        # Get sample id
        sample_id = basename(ct_path).split(".")[0]

        # Read ct image
        sitk_ct = sitk.ReadImage(ct_path)
        ct_arr = sitk.GetArrayFromImage(sitk_ct)

        # Read skeletal muscle segmentation
        sm_path = join(ts_dir, sample_id, "skeletal_muscle.nii.gz")
        sitk_sm = sitk.ReadImage(sm_path)
        sm_arr = sitk.GetArrayFromImage(sitk_sm)

        imat_arr = np.where(sm_arr == 1, ct_arr, 0)
        imat_arr = np.where((imat_arr >= -190) & (imat_arr <= -30), 1, 0)
        imat_arr = imat_arr.astype(np.uint32)

        sitk_out = sitk.GetImageFromArray(imat_arr)
        sitk_out.CopyInformation(sitk_sm)
        sitk.WriteImage(
            sitk_out,
            join(
                join(ts_dir, sample_id),
                f"intermuscular_fat.nii.gz",
            ),
        )


if __name__ == "__main__":
    main()
