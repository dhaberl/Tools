import os
from glob import glob

from natsort import natsorted

if __name__ == "__main__":
    inp_dir = "path/to/ct_dir"
    out_dir = "path/to/out_dir"

    inp_paths = natsorted(glob(os.path.join(inp_dir, "*.nii.gz")))

    for inp_path in inp_paths:
        print(os.path.basename(inp_path))

        dst = os.path.join(out_dir, os.path.basename(inp_path).split(".")[0])
        if not os.path.exists(dst):
            os.makedirs(dst, exist_ok=True)
            print(dst)

        cmd = f"TotalSegmentator -i {inp_path} -o {dst}"
        # cmd = f"TotalSegmentator -i {inp_path} -o {dst} -ta tissue_types"
        # cmd = f"TotalSegmentator -i {inp_path} -o {dst} -ta heartchambers_highres"
        # cmd = (
        #     f"TotalSegmentator -i {inp_path} -o {dst} -rs vertebrae_L1 vertebrae_L5"
        # )

        os.system(cmd)
