from glob import glob
from multiprocessing import Pool
from os.path import basename, join

import pandas as pd
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


if __name__ == "__main__":
    extract_mtv(
        img_dir="/path/to/img_dir/",
        seg_dir="/path/to/seg_dir/",
        save_as="/path/to/out_dir/mtv.csv",
        n_jobs=24,
    )
