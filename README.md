# Cancer-induced vascular changes

Lymph node tumor metastasis is one of the most important prognostic factors in human cancer and correlate with an increased risk for further metastatic spread in several types of cancer, including breast cancer and pancreatic cancer, which are tumor types we focus on in our studies. We have identified cancer-induced vascular changes that can provide value as biomarkers. To evaluate this further we have developed HEV Finder, a new automated image analysis tool that recognizes and characterizes changes of the vasculature in biobank lymph node samples from patients with different types of cancer and invasiveness.

![Graphical abstract](../blob/master/img/graphical_abstract.png?raw=true)

## Code

- HEV Finder is using a mask RCNN deep learning model, based on [BupyeongHealer/Mask_RCNN_tf_2.x](https://github.com/BupyeongHealer/Mask_RCNN_tf_2.x). The detail of the Mask-RCNN implementation can be found in [code/maskrcnn_vessels](https://github.com/BIIFSweden/MariaUlvmar2020-1/tree/master/code/maskrcnn_vessels) and a notebook inspecting the vessel classification in [code/maskrcnn_vessels/samples/vessels/inspect_vessels_model.ipynb](https://github.com/BIIFSweden/MariaUlvmar2020-1/blob/master/code/maskrcnn_vessels/samples/vessels/inspect_vessels_model.ipynb).

- The HEV Finder Graphical User Interface is built on PySimpleGUI and can be found in the [code/HEV_Finder](https://github.com/BIIFSweden/MariaUlvmar2020-1/tree/master/code/HEV_Finder) directory.

- The QuPath script used to import classification into QuPath is located in the [code/QuPath](https://github.com/BIIFSweden/MariaUlvmar2020-1/tree/master/code/QuPath) directory.

## Acknowledgement

This project has support from the BioImage Informatics Facility, a unit of the National Bioinformatics Infrastructure Sweden NBIS, with funding from SciLifeLab, National Microscopy Infrastructure NMI (VR-RFI 2019-00217), and the Chan-Zuckerberg Initiative.
