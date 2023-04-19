outputFolder = "/path/to/out_dir/"

from DICOMLib import DICOMUtils

patientUIDs = slicer.dicomDatabase.patients()
for patientUID in patientUIDs:
    loadedNodeIDs = DICOMUtils.loadPatientByUID(patientUID)
    for loadedNodeID in loadedNodeIDs:
        # Check if we want to save this node
        node = slicer.mrmlScene.GetNodeByID(loadedNodeID)
        # Only export images
        if not node or not node.IsA("vtkMRMLScalarVolumeNode"):
            continue
        # Construct filename
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        seriesItem = shNode.GetItemByDataNode(node)
        studyItem = shNode.GetItemParent(seriesItem)
        patientItem = shNode.GetItemParent(studyItem)
        filename = shNode.GetItemAttribute(patientItem, "DICOM.PatientName")
        # filename += "_" + shNode.GetItemAttribute(studyItem, "DICOM.StudyDate")
        # filename += "_" + shNode.GetItemAttribute(seriesItem, "DICOM.SeriesNumber")
        filename += "_" + shNode.GetItemAttribute(seriesItem, "DICOM.Modality")
        filename = (
            slicer.app.ioManager().forceFileNameValidCharacters(filename) + ".nii.gz"
        )
        # Save node
        print(f"Write {node.GetName()} to {filename}")
        success = slicer.util.saveNode(node, outputFolder + "/" + filename)
    slicer.mrmlScene.Clear()
