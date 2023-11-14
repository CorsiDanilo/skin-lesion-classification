# Melanoma detection

## Installation

We use Python 3.10.11 which is the last version supported by PyTorch. To crete the environment using conda just do

```
conda env create -f environment.yaml
conda activate aml_project
```

## Config

Inside of the `data` folder there should be three elements:

-   `HAM10000_images_train`: the directory containing the train images (Original name of the folder is `HAM10000_images_part_1` and `HAM10000_images_part_2`)
-   `HAM10000_images_test`: the directory containing the test images (original name of the folder is `ISIC2018_Task3_Test_Images`)
-   `HAM10000_segmentations_lesion_tschandl`: the directory containing the masks for the images
-   `HAM10000_metadata_train.csv`: the .csv file containing the metadata for the train images (labels and other info). (Original name of the file is `HAM10000_metadata`)
-   `HAM10000_metadata_test.csv`: the .csv file containing the metadata for the test images (labels and other info). (Original name of the file is `ISIC2018_Task3_Test_GroundTruth.csv`)
