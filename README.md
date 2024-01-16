# Melanoma detection

## Installation

We use Python 3.10.11 which is the last version supported by PyTorch. To crete the environment using conda just do

```
conda env create -f environment.yaml
conda activate aml_project
```

## Data

Inside of the `data` folder there should be three elements:

-   `HAM10000_images_train`: the directory containing the train images (Original name of the folder is `HAM10000_images_part_1` and `HAM10000_images_part_2`)
-   `HAM10000_segmentations_lesion_tschandl`: the directory containing the masks for the images
-   `HAM10000_metadata_train.csv`: the .csv file containing the metadata for the train images (labels and other info). (Original name of the file is `HAM10000_metadata`)
-   `HAM10000_augmented_images`: The directory containing the GAN generated images to augment the training set
-   `synthetic_metadata_train.csv`: The .csv file containing the metadata for the GAN generated images.


Moreover, in order to used SAM it is necessary to put the `sam_checkpoints.pt` file inside the `checkpoints` folder.

## Training
To train a model:
-   Change the configurations you wish to use in the `config.py` file
-   To train a CNN, run `python -m train_loops.CNN_pretrained`. To train a ViT, run `python -m train_loops.ViT`.
-   If you want to resume the training of a model, set `RESUME=True` in the config file, select the folder in which the model is and the checkpoint model. Then start the training again with the command above.

## Testing
To test a model go to `train_loops.test_loop` and select the model to test (chechpoint number or "best"). Then start `python -m train_loops.test_loop`.

## Plots
The plotting function can be found in the `plots` folder. Ensure you have the `tr_val_results.json` and `test_results.json` for a specific model in the `results` folder.
Select the folder name of the model(s) of which to plot the results, then run `python -m plots.test_plots` and `python -m plots.train_plots`.

## Demo
To run the demo, execute `python -m demo.demo_app`. Notice that it is necessary to choose the model to use in the `demo.demo_app` file (the process is the same as the one used in the testing phase).

