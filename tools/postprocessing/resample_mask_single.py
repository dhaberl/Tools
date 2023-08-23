from os.path import basename, dirname, join

import SimpleITK as sitk


def resample_sitk_seg(seg, reference):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetSize(reference.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampled = resampler.Execute(seg)

    return resampled


if __name__ == "__main__":
    path_img = "/path/to/img"
    path_seg = "/path/to/seg"
    sitk_img = sitk.ReadImage(path_img)
    sitk_seg = sitk.ReadImage(path_seg)
    # sitk_out = resample_sitk_seg(sitk_seg, sitk_img)
    sitk_seg.SetDirection(sitk_img.GetDirection())
    sitk_seg.SetOrigin(sitk_img.GetOrigin())
    sitk_seg.SetSpacing(sitk_img.GetSpacing())

    sitk.WriteImage(sitk_seg, join(dirname(path_seg), f"{basename(path_seg).split('.')[0]}_res.nii.gz"))
