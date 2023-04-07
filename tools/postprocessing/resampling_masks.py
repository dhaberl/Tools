from glob import glob
from multiprocessing import Pool
from os import makedirs
from os.path import basename, join

import pandas as pd
import SimpleITK as sitk
from natsort import natsorted


def resample_sitk_mask(image, reference):

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetSize(reference.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampled = resampler.Execute(image)

    return resampled


def mp_resample(output_dir, sample_id, refpet_path, ctmask_dir):
    print(f"Processing: {sample_id}")

    # Make output folder
    makedirs(join(output_dir, sample_id), exist_ok=True)

    ref_pet = sitk.ReadImage(refpet_path)

    masks = natsorted(glob(join(ctmask_dir, "*.nii.gz")))
    for mask in masks:
        sitk_mask = sitk.ReadImage(mask)
        sitk_res = resample_sitk_mask(image=sitk_mask, reference=ref_pet)
        save_as = f"{basename(mask)}.nii.gz"
        sitk.WriteImage(sitk_res, join(output_dir, sample_id, save_as))


def resample_ctmask_to_petmask(csv_path, output_dir, n_jobs=1):
    """
    Resamples the CT-based segmentation masks to a reference image (e.g., PET or SUV)

    CSV-File:
    col1 named "ID" (Unique identifier, Sample ID)
    col2 named "CTMASK_DIR" (Path to directory containing the CT-based masks, e.g. /home/dhaberl/PatientA_CTmasks/)
    col3 named "REFERENCE_PET" (Path to reference image, e.g. /home/dhaberl/PatientA_PET.nii.gz)

    All masks in /home/dhaberl/PatientA_CTmasks/ will be resampled to the reference image.

    Parameters
    ----------
    csv_path : str
        Path to input csv file
    output_dir : str
        Path to output directory, your results will be saved here
    n_jobs : int, optional
        Number of CPU cores to use, by default 1. Dont go crazy, otherwise you run OOM.
    """

    print(f"Executing on {n_jobs} cores.")

    df = pd.read_csv(csv_path)

    mp_args = []
    for i, (ctmask_dir, refpet_path) in enumerate(
        zip(df["CTMASK_DIR"], df["REFERENCE_PET"])
    ):
        sample_id = df["ID"][i]

        mp_args.append((output_dir, sample_id, refpet_path, ctmask_dir))

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        p.starmap(mp_resample, mp_args)


if __name__ == "__main__":
    csv_path = "/path/to/csv/input_resampling.csv"
    output_dir = "/path/to/out_dir"
    n_jobs = 24
    resample_ctmask_to_petmask(csv_path, output_dir, n_jobs)
