import os
from glob import glob
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from natsort import natsorted


def resample_sitk_image(sitk_image, spacing=None, interpolator=None, fill_value=0):
    # https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
    _SITK_INTERPOLATOR_DICT = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "gaussian": sitk.sitkGaussian,
        "label_gaussian": sitk.sitkLabelGaussian,
        "bspline": sitk.sitkBSpline,
        "hamming_sinc": sitk.sitkHammingWindowedSinc,
        "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
        "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
        "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
    }

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = "linear"
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError("Set `interpolator` manually, " "can only infer for 8-bit unsigned or 16, 32-bit signed integers")
        if pixelid == 1:  # 8-bit unsigned int
            interpolator = "nearest"

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing] * num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), "`interpolator` should be one of {}".format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetTransform(sitk.Transform())
    resample_filter.SetDefaultPixelValue(orig_pixelid)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetDefaultPixelValue(fill_value)

    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image


def create_mip(array, axis):
    """Creates MIP along given axis"""
    if axis == "transversal":
        mip = np.max(array, axis=0)
    elif axis == "coronal":
        mip = np.max(array, axis=1)
    elif axis == "sagittal":
        mip = np.max(array, axis=2)
    else:
        print("Uknown axis name:", axis)

    return mip


def make_pet_seg_mip(img_path, mask_path, spacing, view, out_dir, clip_suv, alpha):
    print(f"Processing: {img_path}")

    # Read image and mask
    sitk_img = sitk.ReadImage(img_path)
    sitk_mask = sitk.ReadImage(mask_path)

    # Resample
    sitk_img = resample_sitk_image(sitk_img, spacing=spacing, interpolator="bspline", fill_value=0)
    sitk_mask = resample_sitk_image(sitk_mask, spacing=spacing, interpolator="nearest", fill_value=0)

    # Get npy array from sitk
    img = sitk.GetArrayFromImage(sitk_img)
    img = np.transpose(img)
    mask = sitk.GetArrayFromImage(sitk_mask)
    mask = np.transpose(mask)

    # Clip SUV before making MIP
    if clip_suv:
        img = np.clip(img, 0, clip_suv)

    # Create maximum intensity projection
    mip = create_mip(img, axis=view)
    mask = create_mip(mask, axis=view)

    # Rotate 90°
    mip = np.rot90(mip)
    mask = np.rot90(mask)

    # Mask segmentation
    mask = np.ma.masked_where(mask == 0, mask)

    # Plot
    plt.imshow(mip, cmap="gray_r")
    plt.imshow(mask, alpha=alpha, cmap="coolwarm_r")
    plt.axis("off")
    plt.savefig(
        os.path.join(out_dir, f"{os.path.basename(img_path).split('.')[0]}.png"),
        bbox_inches="tight",
        transparent=False,
        pad_inches=0,
        dpi=200,
    )
    plt.close()


def make_pet_mip(img_path, out_dir, spacing, view, clip_suv):
    print(f"Processing: {img_path}")

    # Read image
    sitk_img = sitk.ReadImage(img_path)

    # Resample
    sitk_img = resample_sitk_image(sitk_img, spacing=spacing, interpolator="bspline", fill_value=0)

    # Get npy array from sitk
    img = sitk.GetArrayFromImage(sitk_img)
    img = np.transpose(img)

    # Clip SUV before making MIP
    if clip_suv:
        img = np.clip(img, 0, clip_suv)

    # Create maximum intensity projection
    mip = create_mip(img, axis=view)

    # Rotate 90°
    mip = np.rot90(mip)

    # Plot
    plt.imshow(mip, cmap="gray_r")

    plt.axis("off")
    plt.savefig(
        os.path.join(out_dir, f"{os.path.basename(img_path).split('.')[0]}.png"),
        bbox_inches="tight",
        transparent=False,
        pad_inches=0,
        dpi=200,
    )
    plt.close()


def make_mip_from_folder(img_dir, out_dir, spacing, view, clip_suv, alpha=None, mask_dir=None, n_jobs=1):
    """Creates maximum intensity projections"""
    img_paths = natsorted(glob(os.path.join(img_dir, "*.nii.gz")))

    mp_args = []
    if mask_dir:
        mask_state = 1
        mask_paths = natsorted(glob(os.path.join(mask_dir, "*.nii.gz")))
        for img_path, mask_path in zip(img_paths, mask_paths):
            mp_args.append((img_path, mask_path, spacing, view, out_dir, clip_suv, alpha))
    else:
        mask_state = 0
        for img_path in img_paths:
            mp_args.append((img_path, out_dir, spacing, view, clip_suv))

    # Assign function to execute
    if mask_state:
        func = make_pet_seg_mip
    else:
        func = make_pet_mip

    # Execute on multiple cores
    with Pool(processes=n_jobs) as p:
        p.starmap(func, mp_args)


if __name__ == "__main__":
    # Example for overlayed PET/SUV-MIP with SEG-MIP
    make_mip_from_folder(
        img_dir="/path/to/suv_dir",
        out_dir="/path/to/out_dir",
        spacing=[2, 2, 2],
        view="coronal",
        clip_suv=8,
        alpha=0.3,
        mask_dir="/path/to/seg_dir",
        n_jobs=16,
    )

    # Example for PET/SUV-MIP
    # make_mip_from_folder(
    #     img_dir="/path/to/suv_dir",
    #     out_dir="/path/to/out_dir",
    #     spacing=[2, 2, 2],
    #     view="coronal",
    #     clip_suv=8,
    #     n_jobs=16,
    # )
