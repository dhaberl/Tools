import glob
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import six
from natsort import natsorted
from radiomics import featureextractor
from skimage.measure import label


class FeatureExtractor:
    def __init__(self, image_dir, segmentation_dir, output_dir) -> None:
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.output_dir = output_dir

    def remove_lesions_below_threshold(self, segmentation_arr, min_lesion_size):
        # Threshold to a binary mask
        segmentation_arr[segmentation_arr != 0] = 1

        # Label individual unique lesions
        # 0 is assumed to be background
        segmentation_arr = label(
            segmentation_arr, background=0, connectivity=3
        )  # default connectivity == input.ndim https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
        lesions = np.unique(segmentation_arr)
        lesions = lesions[lesions > 0]
        # print(f"List of unique lesions: {lesions}")
        lesion_list = []
        for lesion in lesions:
            # print(f"Lesion: {lesion}")
            # Get individual mask for each lesion
            unique_lesion = np.where(segmentation_arr == lesion, lesion, 0)
            # Exclude lesions smaller than min_lesion_size voxels
            unique_lesion_size = np.count_nonzero(unique_lesion)
            # print("size: ", unique_lesion_size)
            if unique_lesion_size < min_lesion_size:
                # print("Lesion too small to be considered. Skipping.")
                pass
            else:
                # Threshold to a binary mask
                unique_lesion[unique_lesion != 0] = 1
                lesion_list.append(unique_lesion)

        return lesion_list

    def prepare_feature_table(self, dataframe, shape_features=False):
        # Remove "diagnostics*" columns
        diagnostics = [c for c in dataframe if not c.startswith("diagnostics")]
        dataframe = dataframe.filter(diagnostics, axis=1)

        if not shape_features:
            # Remove "shape*" columns
            shapes = [c for c in dataframe if not c.startswith("original_shape")]
            dataframe = dataframe.filter(shapes, axis=1)

        return dataframe

    def feature_aggregation(self, features_per_patient, scheme="average"):
        if scheme == "only_largest":
            lesion_size = features_per_patient["original_shape_VoxelVolume"]
            largest_lesion_index = np.argmax(lesion_size)
            features_per_patient = features_per_patient.iloc[largest_lesion_index, :]
            return features_per_patient

        # Shape features are summed
        shape_features_per_patient = features_per_patient.filter(
            [c for c in features_per_patient if c.startswith("original_shape")]
        )
        summed_shape_features = shape_features_per_patient.sum()

        # All other features are averaged
        rest_features_per_patient = features_per_patient.filter(
            [c for c in features_per_patient if not c.startswith("original_shape")]
        )
        if scheme == "weighted_average":
            weights = shape_features_per_patient["original_shape_VoxelVolume"]
            averaged_rest_features = np.average(
                rest_features_per_patient, weights=weights, axis=0
            )
        if scheme == "average":
            averaged_rest_features = np.average(rest_features_per_patient, axis=0)

        averaged_rest_features = pd.Series(
            dict(zip(rest_features_per_patient.keys(), averaged_rest_features))
        )

        # Concat to a single feature vector per patient
        concated = pd.concat([summed_shape_features, averaged_rest_features])

        return concated

    def feature_extraction(self, scheme, min_lesion_size):
        suv_paths = natsorted(glob.glob(os.path.join(self.image_dir, "*")))
        segmentation_paths = natsorted(
            glob.glob(os.path.join(self.segmentation_dir, "*"))
        )
        num_files = len(suv_paths)

        feature_vector_list = []
        for i, (suv_path, segmentation_path) in enumerate(
            zip(suv_paths, segmentation_paths)
        ):
            print(f"{i+1}/{num_files} {suv_path}")
            # Read SUV and SEG
            suv = sitk.ReadImage(suv_path)
            segmentation = sitk.ReadImage(segmentation_path)
            suv_arr = sitk.GetArrayFromImage(suv)
            segmentation_arr = sitk.GetArrayFromImage(segmentation)
            suv_arr = np.transpose(suv_arr)
            segmentation_arr = np.transpose(segmentation_arr)

            # Remove lesions below given threshold
            lesion_list = self.remove_lesions_below_threshold(
                segmentation_arr, min_lesion_size
            )

            # In case no lesion is left
            if not lesion_list:
                print(f"No lesion > {min_lesion_size} voxels left.")
                continue

            features_per_lesion = []
            for j, lesion in enumerate(lesion_list):
                lesion = np.transpose(lesion)
                lesion = lesion.astype(np.uint16)
                lesion = sitk.GetImageFromArray(lesion)
                # Copy metadata
                lesion.SetSpacing(segmentation.GetSpacing())
                lesion.SetDirection(segmentation.GetDirection())
                lesion.SetOrigin(segmentation.GetOrigin())

                # Feature extraction
                extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
                features = extractor.execute(suv, lesion)
                feature_dict = {}
                for key, val in six.iteritems(features):
                    feature_dict[key] = val
                feature_df = pd.Series(feature_dict)
                feature_df = feature_df.to_frame().T
                case_id = os.path.basename(suv_path).split(".")[0]
                case_id = f"{case_id}-L{j+1:02d}"
                feature_df.index = [case_id]
                feature_df.index.name = "ID"

                # Save extracted features per lesion
                features_per_lesion.append(feature_df)

            # Process features per lesion per patient
            features_per_patient = pd.concat(features_per_lesion)
            features_per_patient = self.prepare_feature_table(
                features_per_patient, shape_features=True
            )

            # Per lesion
            if scheme == "per_lesion":
                print(f"Scheme: {scheme}")
                feature_vector_list.append(features_per_patient)
            else:
                # Feature aggregation
                if scheme == "weighted_average":
                    print(f"Scheme: {scheme}")
                    concated = self.feature_aggregation(
                        features_per_patient, scheme=scheme
                    )

                if scheme == "average":
                    print(f"Scheme: {scheme}")
                    concated = self.feature_aggregation(
                        features_per_patient, scheme=scheme
                    )

                if scheme == "only_largest":
                    print(f"Scheme: {scheme}")
                    concated = self.feature_aggregation(
                        features_per_patient, scheme=scheme
                    )

                patient_id = os.path.basename(suv_path).split(".")[0]
                concated = pd.DataFrame(concated.to_dict(), index=[patient_id])
                concated.index.name = "ID"
                feature_vector_list.append(concated)

            # if i+1 >= 10:
            #     break

            print()

        feature_df = pd.concat(feature_vector_list)

        return feature_df


def get_median_spacing(image_dir):
    img_paths = natsorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))

    xs = []
    ys = []
    zs = []
    for img_path in img_paths:
        # print(os.path.basename(img_path))
        sitk_img = sitk.ReadImage(img_path)
        x, y, z = sitk_img.GetSpacing()
        print(round(x), round(y), round(z))
        xs.append(x)
        ys.append(y)
        zs.append(z)

    print(np.median(xs))
    print(np.median(ys))
    print(np.median(zs))


if __name__ == "__main__":
    image_dir = "/path/to/img_dir"  # Path to directory containing images
    segmentation_dir = "/path/to/seg_dir"  # Path to directory containing corresponding lesion segmentations
    output_dir = "/path/to/out_dir"  # Path to output directory
    param_file = "/path/to/params_suv.yaml"  # Path to parameter file

    # TODO: Minimum lesion size should be checked after resampling!
    min_lesion_size = 64  # minimum lesion size in voxels (lesions below this size will not be considered for feature extraction; see: DOI: 10.1016/j.cpet.2021.06.007)

    # get_median_spacing(image_dir)
    feature_extractor = FeatureExtractor(image_dir, segmentation_dir, output_dir)

    # -------------------------------------------------------------------------
    # Scheme how to aggregate the features when multiple lesions are present
    # Similar to: https://www.nature.com/articles/s41598-021-89114-6
    # -------------------------------------------------------------------------

    # Per lesion feature extraction, features are saved per lesion
    # scheme = "per_lesion"

    # Features extracted from each lesion individually
    # Histogram and texture features: unweighted average
    # Size and shape features: Summed
    # scheme = "average"

    # Features extracted from each lesion individually
    # Histogram and texture features: volume weighted average
    # Size and shape features: Summed
    # scheme = "weighted_average"

    # Features extracted from the largest lesion only
    # scheme = "only_largest"

    # Run all schemes
    schemes = ["only_largest", "per_lesion", "average", "weighted_average"]
    for scheme in schemes:
        feature_df = feature_extractor.feature_extraction(
            scheme=scheme, min_lesion_size=min_lesion_size
        )
        feature_df.to_csv(os.path.join(output_dir, f"{scheme}.csv"))
