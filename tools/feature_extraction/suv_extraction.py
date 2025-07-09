from glob import glob
from multiprocessing import Pool
from os.path import basename, join

import pandas as pd
from natsort import natsorted
from qib import FeatureExtractor


def mp_extract(suv_path, masks, sample_id):

    out_dict = {}
    for mask in masks:
        mask_id = basename(mask).split(".")[0]
        print(f"Processing: {sample_id} {mask_id}")

        roi = FeatureExtractor(img_path=suv_path, seg_path=mask)

        suvmean = roi.suvmean()
        suvmin = roi.suvmin()
        suvmax = roi.suvmax()
        suvpeak = roi.suvpeak()

        out_dict["ID"] = [sample_id]
        out_dict[f"{mask_id}_SUVmean"] = [suvmean]
        out_dict[f"{mask_id}_SUVmin"] = [suvmin]
        out_dict[f"{mask_id}_SUVmax"] = [suvmax]
        out_dict[f"{mask_id}_SUVpeak"] = [suvpeak]

    return pd.DataFrame(out_dict)


def extract_suv(csv_path, save_as, n_jobs=1):
    """
    Extracts SUV parameters from the provided segmentation masks as defined in the input csv table.

    CSV-File:
    col1 named "ID" (Unique identifier, Sample ID)
    col2 named "SUV_PATH" (Path to SUV image, e.g. /home/dhaberl/PatientA_SUV.nii.gz)
    col3 named "MASK_DIR" (Path to directory containing the masks, e.g. /home/dhaberl/PatientA_masks/)

    Parameters
    ----------
    csv_path : str
        Path to input csv file
    save_as : str
        Output filename where results will be saved (csv-file)
    n_jobs : int
        Number of CPU cores to use, by default 1. Dont go crazy, otherwise you run OOM.
    """
    print(f"Executing on {n_jobs} cores.")

    df = pd.read_csv(csv_path)

    mp_args = []
    for i, (suv_path, mask_dir) in enumerate(zip(df["SUV_PATH"], df["MASK_DIR"])):
        sample_id = df["ID"][i]
        masks = natsorted(glob(join(mask_dir, "*.nii.gz")))
        mp_args.append((suv_path, masks, sample_id))

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        out_df = p.starmap(mp_extract, mp_args)

    # Concatenate results
    out_df = pd.concat(out_df, ignore_index=True)
    print(out_df)

    # Save
    # out_df.to_csv(save_as, index=False)
    out_df.to_excel(save_as, index=False)


if __name__ == "__main__":
    extract_suv(
        csv_path="/path/to/input_suvextraction.csv",
        save_as="/path/to/output/suv_parameters.csv",
        n_jobs=24,
    )
