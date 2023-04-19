from glob import glob
from os import makedirs
from os.path import basename, join
from shutil import rmtree
from subprocess import call
from uuid import uuid4

import pandas as pd
import pydicom
import SimpleITK as sitk
from natsort import natsorted
from tqdm import tqdm


def convert_from_folder(csv_file, output_dir, path_to_module, save_meta):
    """
    Converts dicom-PET to nifti-PET and calculates nifti-SUV.
    The SUV calculation (specifically: SUVbw, normalized to body weight) is outsourced
    to 3DSlicer/PETDICOMExtension. Dicom conversion is outsourced to dcm2niix.
    """
    # Make directory to save PET niftis
    pet_dir = join(output_dir, "PET_nifti")
    makedirs(pet_dir, exist_ok=True)

    # Make temporary directory to save metadata
    tmp_id = uuid4()
    tmp_dir = join(output_dir, f"_tmp-{tmp_id}")
    makedirs(tmp_dir, exist_ok=True)

    # Read input csv
    df = pd.read_csv(csv_file, dtype=str)

    for i, (sample_id, dicomdir) in enumerate(zip(df["ID"], df["DICOMDIR"])):
        print(f"{i+1}/{len(df)}", sample_id)
        print(f"DICOMDIR: {dicomdir}")

        # Convert dcm PET to nifti PET
        cmd = f"dcm2niix -o {pet_dir} -z y -f {sample_id + '_PET'} {dicomdir}"
        call(cmd, shell=True)

        # Read out SeriesInstanceUID from one dcm file
        query_file = natsorted(glob(join(dicomdir, "*")))[0]
        dcm = pydicom.dcmread(query_file)
        sid = dcm.SeriesInstanceUID
        print(f"Found Modality: {dcm.Modality}")
        print(f"Found SeriesInstanceUID: {dcm.SeriesInstanceUID}")

        # Get SUV conversion factors
        cmd = f"cd {path_to_module}; ./SUVFactorCalculator -p {dicomdir} --petSeriesInstanceUID {sid} -r {tmp_dir} --returnparameterfile {join(tmp_dir, sample_id)}.txt"
        call(cmd, shell=True)
        print()

    # Parse parameter files
    file_paths = natsorted(glob(join(tmp_dir, "*.txt")))

    out_dict = {"ID": [], "SUVbwConversionFactor": []}
    for file_path in file_paths:
        with open(file_path) as f:
            text = f.readlines()

        for line in text:
            if "SUVbwConversionFactor" in line:
                l = line.split("\n")[:-1]
                l = l[0]
                l = l.split(" ")
                suvbw_conversion_factor = float(l[-1])
                out_dict["ID"].append(basename(file_path).split(".")[0])
                out_dict["SUVbwConversionFactor"].append(suvbw_conversion_factor)

    out_df = pd.DataFrame(out_dict)
    out_df.to_csv(join(tmp_dir, "SUVbw_conversion_factors.csv"), index=False)

    # Make directory to save SUV niftis
    suv_dir = join(output_dir, "SUV_nifti")
    makedirs(suv_dir, exist_ok=True)

    # Apply conversion factors
    print("Converting PET images to SUV images")
    for sample_id, suv_factor in tqdm(
        zip(out_df["ID"], out_df["SUVbwConversionFactor"]), total=len(df)
    ):
        # Load nifti PET
        sitk_img = sitk.ReadImage(join(pet_dir, f"{sample_id}_PET.nii.gz"))
        img_arr = sitk.GetArrayFromImage(sitk_img)
        img_arr = img_arr * suv_factor
        sitk_suv = sitk.GetImageFromArray(img_arr)
        sitk_suv.CopyInformation(sitk_img)
        sitk.WriteImage(sitk_suv, join(suv_dir, f"{sample_id}_SUV.nii.gz"))

    if save_meta:
        print("Removing temporary metadata folder")
        rmtree(tmp_dir)


if __name__ == "__main__":
    path_to_module = "/home/dhaberl/Slicer-5.2.2-linux-amd64/NA-MIC/Extensions-31382/PETDICOMExtension/lib/Slicer-5.2/cli-modules"
    csv_file = "/path/to/csv/id_dicomdir.csv"
    output_dir = "/path/to/out_dir"
    save_meta = True
    convert_from_folder(csv_file, output_dir, path_to_module, save_meta)

    # Linking
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/dhaberl/Slicer-5.2.2-linux-amd64/lib/Python/lib
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/dhaberl/Slicer-5.2.2-linux-amd64/lib
    # export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/dhaberl/Slicer-5.2.2-linux-amd64/lib/Slicer-5.2
