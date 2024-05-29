from glob import glob
from os.path import basename, join

import SimpleITK as sitk
from natsort import natsorted

if __name__ == "__main__":
    inp_dir = ""

    inp_paths = natsorted(glob(join(inp_dir, "*.nii")))
    inp_paths = [i for i in inp_paths if i.endswith(".nii")]
    print(inp_paths)

    for index, inp_path in enumerate(inp_paths):
        print(f"{index+1}/{len(inp_paths)} {basename(inp_path)}")
        sitk_img = sitk.ReadImage(inp_path)
        sitk.WriteImage(sitk_img, f"{inp_path}.nii.gz")
