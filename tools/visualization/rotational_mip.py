import glob
from multiprocessing import Pool
from os.path import basename, join

import numpy as np
import SimpleITK as sitk
from natsort import natsorted
from PIL import Image


def rescale_to_any_range(image, a, b):
    new_image = ((image - np.min(image)) * (b - a)) / (np.max(image) - np.min(image))
    return new_image


def make_rotational_mip(filepath, interpolator, is_seg, spacing, output_dir):
    """
    Refer to:
    https://discourse.itk.org/t/generate-rotational-maximum-intensity-projections/3226/10
    """
    sample_id = basename(filepath).split(".")[0]

    image = sitk.ReadImage(filepath)
    # print(f"Image shape: {image.GetSize()}")

    projection = {
        "sum": sitk.SumProjection,
        "mean": sitk.MeanProjection,
        "std": sitk.StandardDeviationProjection,
        "min": sitk.MinimumProjection,
        "max": sitk.MaximumProjection,
    }
    ptype = "max"
    paxis = 0

    rotation_axis = [0, 0, 1]
    rotation_angles = np.linspace(0.0, 2 * np.pi, int(360.0 / 10))
    rotation_center = image.TransformContinuousIndexToPhysicalPoint([(index - 1) / 2.0 for index in image.GetSize()])
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_center)

    # Compute bounding box of rotating volume and the resampling grid structure
    image_indexes = list(zip([0, 0, 0], [sz - 1 for sz in image.GetSize()]))
    image_bounds = []
    for i in image_indexes[0]:
        for j in image_indexes[1]:
            for k in image_indexes[2]:
                image_bounds.append(image.TransformIndexToPhysicalPoint([i, j, k]))

    all_points = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)
        all_points.extend([rotation_transform.TransformPoint(pnt) for pnt in image_bounds])

    all_points = np.array(all_points)
    min_bounds = all_points.min(0)
    max_bounds = all_points.max(0)
    # resampling grid will be isotropic so no matter which direction we project to
    # the images we save will always be isotropic (required for image formats that
    # assume isotropy - jpg,png,tiff...)
    if spacing == "auto":
        new_spc = [np.min(image.GetSpacing())] * 3
    else:
        new_spc = spacing

    # Calculate new size
    new_sz = [int(sz / spc + 0.5) for spc, sz in zip(new_spc, max_bounds - min_bounds)]

    proj_images = []
    for angle in rotation_angles:
        rotation_transform.SetRotation(rotation_axis, angle)
        resampled_image = sitk.Resample(
            image1=image,
            size=new_sz,
            transform=rotation_transform,
            interpolator=interpolator,
            outputOrigin=min_bounds,
            outputSpacing=new_spc,
            outputDirection=[1, 0, 0, 0, 1, 0, 0, 0, 1],
            defaultPixelValue=0,  # -1000, #HU unit for air in CT, possibly set to 0 in other cases
            outputPixelType=image.GetPixelID(),
        )

        proj_image = projection[ptype](resampled_image, paxis)
        extract_size = list(proj_image.GetSize())
        extract_size[paxis] = 0
        proj_images.append(sitk.Extract(proj_image, extract_size))

    # Stack all images into fuax-volume for display
    # sitk.Show(sitk.JoinSeries(proj_images), debugOn=True)

    frames = []
    for proj_image in proj_images:
        image = sitk.GetArrayFromImage(proj_image)
        image = np.flipud(image)
        if not is_seg:
            image = np.clip(image, a_min=0, a_max=np.percentile(image, 99.5))

        # To png and gif
        image = rescale_to_any_range(image, a=0, b=255)
        image = image.astype(np.uint8())
        pil_image = Image.fromarray(image)

        # Save png
        frames.append(pil_image)

    frame_one = frames[0]
    frame_one.save(join(output_dir, f"{sample_id}.gif"), format="GIF", append_images=frames, save_all=True, duration=300, loop=0)
    print(f"Done with {sample_id}")


def generate_from_folder(image_dir, output_dir, is_seg=False, spacing="auto", n_jobs=2):
    image_paths = natsorted(glob.glob(join(image_dir, "*.nii*")))

    mp_args = []
    for image_path in image_paths:
        if is_seg:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkBSpline
        mp_args.append((image_path, interpolator, is_seg, spacing, output_dir))

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        p.starmap(make_rotational_mip, mp_args)


if __name__ == "__main__":
    image_dir = "/media/dhaberl/T7/PhD_Projects/Holger_Kilian_Prostate_Cancer_Data/pre_post_Lu_MCRPC_ordered_output_from_David/SUV_nifti"
    output_dir = "/media/dhaberl/T7/PhD_Projects/Holger_Kilian_Prostate_Cancer_Data/pre_post_Lu_MCRPC_ordered_output_from_David/MIP_rotational"

    generate_from_folder(image_dir, output_dir, is_seg=False, spacing=[2, 2, 2], n_jobs=32)
