from itertools import combinations

import numpy as np
import SimpleITK as sitk
from skimage.measure import label
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self, img_path, seg_path):
        """
        Feature extractor class to calculate common quantitative imaging biomarkers.

        Parameters
        ----------
        img_path : str
            Path to SUV image (.nii.gz)
        seg_path : str
            Path to binary segmentation mask (.nii.gz). Assumes Background: 0, Foreground: 1
        """
        self.sitk_img, self.img = self._read_nifti(img_path)
        self.sitk_seg, self.seg = self._read_nifti(seg_path)

        # Check for empty segmenation files
        self.empty = self._check_empty(self.seg)

    def _check_empty(self, seg):
        """
        Checks if provided segmentation mask is empty or not
        """
        if np.all(seg == 0):
            print("Warning: Your segmentation file is empty. Extraction not possible.")
            return True
        else:
            return False

    def _read_nifti(self, img_path):
        sitk_img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(sitk_img)
        img = np.transpose(img)
        return sitk_img, img

    def suvmean(self):
        """
        Calculates SUVmean
        """
        if self.empty:
            return np.nan

        roi = self.img[self.seg == 1]
        suvmean = np.mean(roi)
        return suvmean

    def suvmin(self):
        """
        Calculates SUVmin
        """
        if self.empty:
            return np.nan

        roi = self.img[self.seg == 1]
        suvmin = np.min(roi)
        return suvmin

    def suvmax(self):
        """
        Calculates SUVmax
        """
        if self.empty:
            return np.nan

        roi = self.img[self.seg == 1]
        suvmax = np.max(roi)
        return suvmax

    def suvvar(self):
        """
        Calculates SUV variance
        """
        if self.empty:
            return np.nan

        roi = self.img[self.seg == 1]
        suvvar = np.var(roi)
        return suvvar

    def suvrange(self):
        """
        Calculates SUV range
        """
        if self.empty:
            return np.nan

        suvrange = self.suvmax() - self.suvmin()
        return suvrange

    def suvsum(self):
        """
        Calculates SUV sum
        """
        if self.empty:
            return np.nan

        roi = self.img[self.seg == 1]
        suvsum = np.sum(roi)
        return suvsum

    def suvpeak(self):
        """
        Calculates SUVpeak
        TODO: This func should be re-written following a more standardized SUVpeak calculation
        """
        if self.empty:
            return np.nan

        # Mask the image array to the segmentation
        # All voxels of where SEG: 1 are kept, all voxels where SEG: 0 are set to zero
        suv_masked = np.where(self.seg == 1, self.img, 0)

        # Get SUVmax
        suvmax = self.suvmax()

        # TODO: What if multiple max? average?
        # print(f"Location of SUVmax: [i, j, k] = {[i[0] for i in np.where(suv_masked == SUVmax)]}")

        # Get coordinates of SUVmax
        i, j, k = np.where(suv_masked == self.suvmax())

        # Calculate SUV peak (using 26 neighbors around SUVmax)
        my_neighbors = self._get_neighbours(i[0], j[0], k[0])

        my_neighbors_values = []
        for i, j, k in my_neighbors:
            if k >= self.img.shape[-1]:
                continue
            my_neighbors_values.append(suv_masked[i, j, k])
        my_neighbors_values = np.array(my_neighbors_values)

        # Remove zeros because they are outside the actual VOI
        suvpeak_bg = my_neighbors_values[my_neighbors_values > 0]

        # Add SUVmax
        suvpeak = np.append(suvmax, suvpeak_bg)
        suvpeak = np.mean(suvpeak)

        return suvpeak

    def mtv(self):
        """
        Calculates the metabolic tumor volume in ml = cm3
        """
        if self.empty:
            return 0

        # Count number of voxels which are non-zero
        num_nonzero_voxels = np.count_nonzero(self.seg)
        # print(f"Number of nonzero voxels: {num_nonzero_voxels}")

        # Calculate volume of a single voxel in mm3
        voxel_vol = np.product(self.sitk_seg.GetSpacing())
        # print(f"Volume of a single voxel (mm3): {voxel_vol}")

        # Calculate the volume of all non-zero voxels (=metabolic tumor volume)
        nonzero_voxel_vol = num_nonzero_voxels * voxel_vol

        # Convert to cm3 = ml
        nonzero_voxel_vol = nonzero_voxel_vol / 1000
        # print(f"Nonzero voxel volume (cm3 = ml): {nonzero_voxel_vol}")

        return nonzero_voxel_vol

    def tlg(self):
        """
        Calculates the total lesion glycolysis
        """
        if self.empty:
            return np.nan

        tlg = self.mtv() * self.suvmean()
        return tlg

    def from_list(self, metrics):
        metric_dict = {}
        for metric in metrics:
            if metric == "suvmean":
                suvmean = self.suvmean()
                metric_dict[metric] = suvmean
            elif metric == "suvmin":
                suvmin = self.suvmin()
                metric_dict[metric] = suvmin
            elif metric == "suvmax":
                suvmax = self.suvmax()
                metric_dict[metric] = suvmax
            elif metric == "suvpeak":
                suvpeak = self.suvpeak()
                metric_dict[metric] = suvpeak
            elif metric == "suvvar":
                suvvar = self.suvvar()
                metric_dict[metric] = suvvar
            elif metric == "suvrange":
                suvrange = self.suvrange()
                metric_dict[metric] = suvrange
            elif metric == "suvsum":
                suvsum = self.suvsum()
                metric_dict[metric] = suvsum
            elif metric == "mtv":
                mtv = self.mtv()
                metric_dict[metric] = mtv
            elif metric == "tlg":
                tlg = self.tlg()
                metric_dict[metric] = tlg
            elif metric == "dmax":
                dmax = self.dmax()
                metric_dict[metric] = dmax
            elif metric == "distance_suvmax_centroid":
                distance_suvmax_centroid = self.distance_suvmax_centroid()
                metric_dict[metric] = distance_suvmax_centroid

        return metric_dict

    def _get_neighbours(self, ic, jc, kc):
        neighbors = []  # initialize the empty neighbor list
        # Find the 26 neighboring voxels' coordinates
        for i in [-1, 0, 1]:  # i coordinate
            for j in [-1, 0, 1]:  # j coordinate
                for k in [-1, 0, 1]:  # k coordinate
                    if i == 0 and j == 0 and k == 0:  # if at the same point
                        pass  # skip the current point
                    else:
                        a = ic + i
                        b = jc + j
                        c = kc + k
                        neighbors.append(
                            (a, b, c)
                        )  # add the neighbor to the neighbors list

        return neighbors

    def dmax(self):
        """
        Calculates the tumor dissemination feature dmax, defined as the largest distance among all lesions (in millimetre mm).
        The pairwise distances of all lesions are computed and the largest distance is returned
        Uses the centroid to define the lesion coordinates
        """
        if self.empty:
            return np.nan

        # Label individual lesions
        # BG: 0, Lesion1: 1, Lesion2: 2, ...
        # Separation of lesions is based on connectivity profile
        # default connectivity == input.ndim https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
        self.seg = label(self.seg, background=0, connectivity=3)

        enumerated_lesions = np.unique(self.seg)
        lesion_ids = enumerated_lesions[enumerated_lesions > 0]
        # If only a single lesion, Dmax is zero
        if len(lesion_ids) == 1:
            print("Only one lesion...")
            return 0
        else:
            # print(lesion_ids)
            pairwise_combinations = list(combinations(lesion_ids, 2))
            # print(pairwise_combinations)
            pairwise_distances = []
            print("Calculating pairwise distances...")
            for combination in tqdm(pairwise_combinations):
                # print(combination)
                l1 = combination[0]
                l2 = combination[1]
                l1_arr = np.where(self.seg == l1, l1, 0)
                l2_arr = np.where(self.seg == l2, l2, 0)
                d = self._pairwise_lesion_distance(
                    l1_arr, l2_arr, self.sitk_seg.GetSpacing()
                )
                pairwise_distances.append(d)
            # print(pairwise_distances)
            maximum_distance = np.max(pairwise_distances)
            return maximum_distance

    def _pairwise_lesion_distance(self, mask1_arr, mask2_arr, spacing):
        """
        Calculate distance between two lesions based on the centroid
        """
        # Calculate centroid for each lesion
        x1, y1, z1 = self._centroid(mask1_arr)
        x2, y2, z2 = self._centroid(mask2_arr)
        # print("Centroid 1:", x1, y1, z1)
        # print("Centroid 2:", x2, y2, z2)

        # Calculate distance
        d = self._two_point_distance(x1, y1, z1, x2, y2, z2, spacing)

        return d

    def _centroid(self, mask_arr):
        x, y, z = (mask_arr != 0).nonzero()
        xmedian = np.median(x)
        ymedian = np.median(y)
        zmedian = np.median(z)
        return (xmedian, ymedian, zmedian)

    def _two_point_distance(self, x1, y1, z1, x2, y2, z2, spacing):
        """
        Calculate distance between two points
        """
        x_sp = spacing[0]
        y_sp = spacing[1]
        z_sp = spacing[2]
        d2 = (
            np.square((x1 - x2) * x_sp)
            + np.square((y1 - y2) * y_sp)
            + np.square((z1 - z2) * z_sp)
        )
        d = np.sqrt(d2)

        return d

    def _enumerate_lesions(self, mask_arr):
        """
        Label individual lesions
        BG: 0, Lesion1: 1, Lesion2: 2, ...
        Separation of lesions is based on connectivity
        Default connectivity == input.ndim https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
        """

        enum_mask_arr = label(mask_arr, background=0, connectivity=3)
        enumerated_lesions = np.unique(enum_mask_arr)
        lesion_ids = enumerated_lesions[enumerated_lesions > 0]

        return lesion_ids, enum_mask_arr

    def distance_suvmax_centroid(self):
        """
        Calculates the distance between the centroid of the lesion and the SUVmax
        """
        if self.empty:
            return np.nan

        # Mask the image array to the segmentation
        # All voxels of where SEG: 1 are kept, all voxels where SEG: 0 are set to zero
        suv_masked = np.where(self.seg == 1, self.img, 0)

        # Get coordinates of SUVmax
        # TODO: What if multiple max?
        i, j, k = np.where(suv_masked == self.suvmax())
        # print(i, j, k)

        # Get coordinates of centroid
        cx, cy, cz = self._centroid(self.seg)
        # print(cx, cy, cz)

        # Calculate distance between the two points
        d = self._two_point_distance(i, j, k, cx, cy, cz, self.sitk_seg.GetSpacing())
        d = d[0]

        return d


def SDmax(Dmax, weight, height):
    """
    Calculates the standardized tumor dissemination feature SDmax.
    Defined as: Dmax normalized by body surface area.

    Parameters
    ----------
    Dmax : float
        Tumor dissemination in mm
    weight : float
        Body weight in kg
    height : foat
        Patient height in cm

    Returns
    -------
    float
        Standardized tumor dissemination in 1/m
    """
    # Normalize to BSA
    bsa = np.sqrt(weight * height / 3600)  # m^2
    print("BSA: ", bsa)
    Dmax_m = Dmax * 0.001  # m
    sdmax = Dmax_m / bsa  # 1/m
    print("SDmax (1/m): ", sdmax)
    return sdmax


def test():
    img_path = "/path/to/img.nii.gz"
    seg_path = "/path/to/seg.nii.gz"
    roi = FeatureExtractor(img_path, seg_path)
    print(roi.suvmean())
    print(roi.suvpeak())
    print(roi.suvmin())
    print(roi.suvmax())
    print(roi.mtv())
    print(roi.tlg())
    print(roi.dmax())
    print(roi.distance_suvmax_centroid())
    print(
        roi.from_list(
            metrics=[
                "suvmean",
                "suvpeak",
                "suvmin",
                "suvmax",
                "mtv",
                "tlg",
                "dmax",
                "distance_suvmax_centroid",
            ]
        )
    )


def run_from_directory():
    from glob import glob
    from os.path import basename, join

    import pandas as pd
    from natsort import natsorted

    img_dir = "path/to/img_dir"
    seg_dir = "path/to/seg_dir"

    img_paths = natsorted(glob(join(img_dir, f"*.nii*")))
    seg_paths = natsorted(glob(join(seg_dir, f"*.nii*")))

    l = []
    for i, (img_path, seg_path) in enumerate(zip(img_paths, seg_paths)):
        print(basename(img_path), basename(seg_path))
        roi = FeatureExtractor(img_path, seg_path)
        d = roi.from_list(
            metrics=[
                "suvmax",
                "suvmean",
                "suvmin",
                "suvrange",
                "suvsum",
                "suvvar",
                "suvpeak",
                "mtv",
            ]
        )
        d["ID"] = basename(img_path).split(".")[0]

        l.append(pd.DataFrame(d, index=[i]))

    out_df = pd.concat(l)

    columns = out_df.columns.tolist()
    new_columns = [columns[-1]] + columns[:-1]

    # Reorder columns in the DataFrame
    out_df = out_df[new_columns]
    out_df.to_csv(
        "suv_parameters.csv",
        index=False,
    )


if __name__ == "__main__":
    run_from_directory()
