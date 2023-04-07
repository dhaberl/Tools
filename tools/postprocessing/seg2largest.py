from glob import glob
from os.path import basename, join

import SimpleITK as sitk
from natsort import natsorted
from tqdm import tqdm


def main():
    """Main"""
    output_dir = "/path/to/out_dir"
    seg_dir = "/path/to/seg_dir"
    seg_paths = natsorted(glob(join(seg_dir, "*.nii.gz")))

    for seg_path in tqdm(seg_paths):
        sitk_img = sitk.ReadImage(seg_path)
        component_image = sitk.ConnectedComponent(sitk_img)
        sorted_component_image = sitk.RelabelComponent(
            component_image, sortByObjectSize=True
        )
        largest_component_binary_image = sorted_component_image == 1
        sitk.WriteImage(
            largest_component_binary_image,
            join(output_dir, f"{basename(seg_path)}.nii.gz"),
        )


if __name__ == "__main__":
    main()
