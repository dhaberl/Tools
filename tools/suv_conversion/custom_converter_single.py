from datetime import datetime
from os.path import join

import numpy as np
import pandas as pd
import SimpleITK as sitk


def assert_time_format(time):
    """
    Time stamp formatting
    Args:
        time (str): Time stamp from DICOM file.
    Returns:
        time: datetime object
    """
    # Cut off milliseconds
    time = time.split(".")[0]
    time_format = "%H%M%S"
    time = datetime.strptime(time, time_format)

    return time


def compute_suvbw_map(img, weight, scan_time, injection_time, half_life, injected_dose):
    """
    Compute SUVbw map based on given weight and injected dose decay.
    Args:
        img: Input image ndarray. Each pixel/voxel is associated with its radioactivity
        represented as volume concentration MBq/mL.
        weight: Patient body weight in kilograms.
        scan_time (str): Acquisition time (start time of PET). Time stamp from DICOM file.
        injection_time (str): Injection time; time when radiopharmaceutical dose was administered.
        Time stamp from DICOM file.
        half_life: Half life of used radiopharmaceutical in seconds.
        injected_dose: Injected total dose of administered radiopharmaceutical in Mega Becquerel.
    Returns:
        suv_map: Image ndarray. Each pixel/voxel is associated with its SUVbw.
    """

    # Assert time format
    scan_time = assert_time_format(scan_time)
    injection_time = assert_time_format(injection_time)
    # Calculate time in seconds between acqusition time (scan time) and injection time
    time_difference = scan_time - injection_time
    time_difference = time_difference.seconds

    # Ensure parameter validity
    check = [weight, time_difference, half_life, injected_dose]
    for i in check:
        assert i > 0, f"Invalid input. No negative values allowed. Value: {i}"
        assert (
            np.isnan(i) == False
        ), f"Invalid input. No NaNs allowed. Value is NaN: {np.isnan(i)}"

    assert weight < 1000, "Weight exceeds 1000 kg, did you really used kg unit?"

    img = np.asarray(img)

    # Calculate decay for decay correction
    decay = np.exp(-np.log(2) * time_difference / half_life)
    # Calculate the dose decayed during procedure in Bq
    injected_dose_decay = injected_dose * decay

    # Weight in grams
    weight = weight * 1000

    # Calculate SUVbw
    suv_map = img * weight / injected_dose_decay

    return suv_map


def convert_single(path_to_csv, output_dir):
    df = pd.read_csv(path_to_csv)

    for i, pet_path in enumerate(df["pet_path"]):
        sitk_img = sitk.ReadImage(pet_path)
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_arr = np.transpose(img_arr)

        weight = df["weight"][i]
        scan_time = str(df["scan_time"][i])
        injection_time = str(df["injection_time"][i])
        half_life = df["half_life"][i]
        injected_dose = df["injected_dose"][i]
        suv_arr = compute_suvbw_map(
            img_arr, weight, scan_time, injection_time, half_life, injected_dose
        )

        suv_arr = np.transpose(suv_arr)
        sitk_out = sitk.GetImageFromArray(suv_arr)
        sitk_out.SetDirection(sitk_img.GetDirection())
        sitk_out.SetOrigin(sitk_img.GetOrigin())
        sitk_out.SetSpacing(sitk_img.GetSpacing())
        sitk.WriteImage(
            sitk_out,
            join(output_dir, f'{df["ID"][i]}_SUV.nii.gz'),
        )


if __name__ == "__main__":
    convert_single(
        path_to_csv="path/to/suv_eda.csv",
        output_dir="path/to/out_dir",
    )
