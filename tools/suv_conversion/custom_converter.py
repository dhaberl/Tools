from datetime import datetime
from glob import glob
from os import makedirs
from os.path import join
from subprocess import call

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from natsort import natsorted


class SUVConerter:
    def __init__(self, batch_file, sep=","):
        self.batch_file = pd.read_csv(batch_file, sep=sep)
        self.ids = self.batch_file["ID"]
        self.dicom_dir = self.batch_file["DICOMDIR"]

    def _get_dicom_tags(self, dcm):
        """
        Return informative and required DICOM tags for SUV calculation. Missing DICOM tags will be returned as NaNs.
        Note: sex and age is not required but can help for estimations if values are missing (e.g. body weight)
        DICOM tags:
        https://dicom.innolitics.com/ciods
        Args:
            dcm (pydicom.dataset.FileDataset): Loaded DICOM file.
            Example:
                dcm = pydicom.dcmread(path_to_dcm_file)
            pydicom:
            https://pydicom.github.io/pydicom/stable/old/ref_guide.html
        Returns:
            dict: Dictionary with DICOM tags.
        """

        # Ensure input parameter validity
        assert (
            dcm.Modality == "PT"
        ), "Passed DICOM file is not a Positron-Emission-Tomography scan. Check DICOM Modality tag."

        # Get patient age
        try:
            age = dcm.PatientAge
        except AttributeError:
            print("Age is not stored in DICOM file.")
            age = np.nan

        # Get patient sex
        try:
            sex = dcm.PatientSex
        except AttributeError:
            print("Sex is not stored in DICOM file.")
            sex = np.nan

        # Get patient weight
        try:
            weight = dcm.PatientWeight
        except AttributeError:
            print("Weight is not stored in DICOM file.")
            weight = np.nan

        # Get patient height
        try:
            patient_height = dcm.PatientSize
        except AttributeError:
            print("Patient Size is not stored in DICOM file.")
            patient_height = np.nan

        # Get radiopharmaceutical information (radiotracer)
        try:
            tracer = dcm.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical
        except AttributeError:
            print("Radiopharmaceutical Info is not stored in DICOM file.")
            tracer = np.nan

        # Get scan time
        try:
            scan_time = dcm.AcquisitionTime
        except AttributeError:
            print("Acquisition Time is not stored in DICOM file.")
            scan_time = np.nan

        # Get start time of the radiopharmaceutical injection
        try:
            injection_time = dcm.RadiopharmaceuticalInformationSequence[
                0
            ].RadiopharmaceuticalStartTime
        except AttributeError:
            print("Injection Time is not stored in DICOM file.")
            injection_time = np.nan

        # Get half life of radionuclide
        try:
            half_life = dcm.RadiopharmaceuticalInformationSequence[
                0
            ].RadionuclideHalfLife
        except AttributeError:
            print("Half Life is not stored in DICOM file.")
            half_life = np.nan

        # Get total dose injected for radionuclide
        try:
            injected_dose = dcm.RadiopharmaceuticalInformationSequence[
                0
            ].RadionuclideTotalDose
        except AttributeError:
            print("Injected Dose is not stored in DICOM file.")
            injected_dose = np.nan

        return {
            "age": [age],
            "sex": [sex],
            "weight": [weight],
            "height": [patient_height],
            "tracer": [tracer],
            "scan_time": [scan_time],
            "injection_time": [injection_time],
            "half_life": [half_life],
            "injected_dose": [injected_dose],
        }

    def _get_query_file(self, dicom_dir):
        # Read out data from one dcm file (=query file)
        # The query file must be a PT file
        all_files = natsorted(glob(join(dicom_dir, "*")))
        dcm = pydicom.dcmread(all_files[0])

        # Check if query file is PT file
        if not dcm.Modality == "PT":
            # print(f"Query file is {dcm.Modality}! Checking another one!")
            for f in all_files:
                dcm = pydicom.dcmread(f)
                if dcm.Modality == "PT":
                    # print(f"Query file is {dcm.Modality}! Sucsess!")
                    break
        else:
            # print(f"Query file is {dcm.Modality}! Sucsess!")
            pass
        return dcm

    def inspect_data(self, save_as=None):
        out_df = []
        for id, dicom_dir in zip(self.ids, self.dicom_dir):
            print(f"Inspecting data: {id}")
            query_file = self._get_query_file(dicom_dir)
            # Read out dicom tags
            dicom_tags = self._get_dicom_tags(query_file)
            dicom_tags["ID"] = [id]
            out_df.append(pd.DataFrame(dicom_tags))
            print()

        df = pd.concat(out_df, ignore_index=True)
        # Reorder so "ID" is first column
        columns = df.columns.to_list()
        columns = columns[-1:] + columns[:-1]
        df = df[columns]
        print(df)

        if save_as:
            df.to_csv(save_as, index=False)

    def _dcm2niix(self, id, input_path, output_path):
        cmd = f"dcm2niix -o {output_path} -z y -f {id + '_PET'} {input_path}"
        call(cmd, shell=True)

    def convert_pet(self, output_dir):
        for id, dicom_dir in zip(self.ids, self.dicom_dir):
            print(id)
            # Convert dcm PET to nifti PET
            self._dcm2niix(id, dicom_dir, output_dir)
            print()

        print("Finished PET conversion!")

    def convert_suv(self, output_dir, half_life=None):
        # Make directory to save PET
        pet_dir = join(output_dir, "PET_nifti")
        makedirs(pet_dir, exist_ok=True)

        # Make directory to save PET
        suv_dir = join(output_dir, "SUV_nifti")
        makedirs(suv_dir, exist_ok=True)

        flg_missing = {"ID": [], "tag": []}
        for id, dicom_dir in zip(self.ids, self.dicom_dir):
            print(f"ID: {id}")

            # Convert dcm PET to nifti PET
            self._dcm2niix(id, dicom_dir, pet_dir)

            # Read out dicom tags
            query_file = self._get_query_file(dicom_dir)
            dicom_tags = self._get_dicom_tags(query_file)
            weight = dicom_tags["weight"][0]
            scan_time = dicom_tags["scan_time"][0]
            injection_time = dicom_tags["injection_time"][0]
            if not half_life:
                half_life = dicom_tags["half_life"][0]
            injected_dose = dicom_tags["injected_dose"][0]

            # Check for missing tags
            skip = 0
            for key, val in zip(
                ["weight", "scan_time", "injection_time", "half_life", "injected_dose"],
                [weight, scan_time, injection_time, half_life, injected_dose],
            ):
                if (
                    val is np.nan
                ):  # ghetto solution, but dont know how to do this properly, be careful with missing dicom tags and inspect your data first (=> inspect_data function)
                    print(f'Tag: "{key}" is missing. Cannot compute SUV. Skipping.')
                    flg_missing["ID"].append(id)
                    flg_missing["tag"].append(key)
                    skip = 1
            if skip:
                continue

            print(f"weight[kg]: {weight}")
            print(f"scan_time[timestamp]: {scan_time}")
            print(f"injection_time[timestamp]: {injection_time}")
            print(f"half_life[s]: {half_life}")
            print(f"injected_dose[MBq]: {injected_dose}")

            # Read nifti PET
            sitk_img = sitk.ReadImage(join(pet_dir, f"{id}_PET.nii.gz"))
            img_arr = sitk.GetArrayFromImage(sitk_img)
            img_arr = np.transpose(img_arr)

            # Compute SUV map
            suv_arr = self._compute_suvbw_map(
                img_arr, weight, scan_time, injection_time, half_life, injected_dose
            )

            # Back to sitk object and save
            suv_arr = np.transpose(suv_arr)
            sitk_out = sitk.GetImageFromArray(suv_arr)
            sitk_out.CopyInformation(sitk_img)
            sitk.WriteImage(sitk_out, join(suv_dir, f"{id}_SUV.nii.gz"))

            print()

    def _assert_time_format(self, time):
        """
        Time stamp formatting
        Args:
            time (str): Time stamp from DICOM file.
        Returns:
            time: datetime object
        """
        # Cut off milliseconds
        time = time.split(".")[0]
        time_format = "%H%M%S"
        time = datetime.strptime(time, time_format)

        return time

    def _compute_suvbw_map(
        self, img, weight, scan_time, injection_time, half_life, injected_dose
    ):
        """
        Compute SUVbw map based on given weight and injected dose decay.
        Args:
            img: Input image ndarray. Each pixel/voxel is associated with its radioactivity
            represented as volume concentration MBq/mL.
            weight: Patient body weight in kilograms.
            scan_time (str): Acquisition time (start time of PET). Time stamp from DICOM file.
            injection_time (str): Injection time; time when radiopharmaceutical dose was administered.
            Time stamp from DICOM file.
            half_life: Half life of used radiopharmaceutical in seconds.
            injected_dose: Injected total dose of administered radiopharmaceutical in Mega Becquerel.
        Returns:
            suv_map: Image ndarray. Each pixel/voxel is associated with its SUVbw.
        """

        # Assert time format
        scan_time = self._assert_time_format(scan_time)
        injection_time = self._assert_time_format(injection_time)
        # Calculate time in seconds between acqusition time (scan time) and injection time
        time_difference = scan_time - injection_time
        time_difference = time_difference.seconds

        # Ensure parameter validity
        check = [weight, time_difference, half_life, injected_dose]
        for i in check:
            assert i > 0, f"Invalid input. No negative values allowed. Value: {i}"
            assert (
                np.isnan(i) == False
            ), f"Invalid input. No NaNs allowed. Value is NaN: {np.isnan(i)}"

        assert weight < 1000, "Weight exceeds 1000 kg, did you really used kg unit?"

        img = np.asarray(img)

        # Calculate decay for decay correction
        decay = np.exp(-np.log(2) * time_difference / half_life)
        # Calculate the dose decayed during procedure in Bq
        injected_dose_decay = injected_dose * decay

        # Weight in grams
        weight = weight * 1000

        # Calculate SUVbw
        suv_map = img * weight / injected_dose_decay

        return suv_map


if __name__ == "__main__":
    x = SUVConerter("/path/to/id_dicomdir.csv")

    # Inspect your data (reads out dicom tags)
    x.inspect_data()

    # Converts dicom PET to nifti PET
    # x.convert_pet(output_dir="/path/to/out_dir")

    # Converts dicom PET to nifti PET and nifti SUV
    # x.convert_suv(output_dir="/path/to/out_dir")
