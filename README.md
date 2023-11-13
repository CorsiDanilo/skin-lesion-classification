# Melanoma detection

## Installation

We use Python 3.10.11 which is the last version supported by PyTorch. To crete the environment using conda just do

```
conda env create -f environment.yaml
conda activate aml_project
```

## Config

Inside of the `data` folder there should be three elements:

-   `HAM10000_images`: the directory containing all the images
-   `HAM10000_segmentations_lesion_tschandl`: the directory containing the masks for the images
-   `HAM10000_metadata.csv`: the .csv file containing the metadata (labels and other info) for each image
