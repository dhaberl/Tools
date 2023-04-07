from glob import glob
from os.path import basename, join

import pandas as pd
from natsort import natsorted


def make_csv(img_dir, mask_dir, save_as):
    """Creates PyRadiomics conform input csv file"""
    img_paths = natsorted(glob(join(img_dir, "*.nii.gz")))
    seg_paths = natsorted(glob(join(mask_dir, "*.nii.gz")))

    out_dict = {"ID": [], "Image": [], "Mask": []}
    for img_path, seg_path in zip(img_paths, seg_paths):
        print(basename(img_path), basename(seg_path))
        out_dict["ID"].append(basename(img_path).split(".")[0])
        out_dict["Image"].append(img_path)
        out_dict["Mask"].append(seg_path)
    out_df = pd.DataFrame(out_dict)
    print(out_df)
    out_df.to_csv(save_as, index=False)


if __name__ == "__main__":
    make_csv(
        img_dir="/path/to/img_dir",
        mask_dir="/path/to/seg_dir",
        save_as="/path/to/output/image_mask.csv",
    )
