# Skin Lesion Classification

This includes the code for different Skin Lesion Classification models to classify the 7 different types of skin lesions using the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

The code was developed partly for the Advanced Machine Learning (2023-2024) and for the Computer Vision (2023-2024) course in the Master Program in Computer Science at the University of Rome "La Sapienza".

The code includes a re-implementation of the [MSLANet - Multi-Scale Long Attention Network for Skin Classification](https://link.springer.com/article/10.1007/s10489-022-03320-x) paper, with some adjustments in order to enhance the performances, along with other solutions using Semantic Segmentation with [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and classification with CNNs and Vision Transformers.

The model also takes some parts of the code from the [python StyleGAN](https://github.com/huangzh13/StyleGAN.pytorch), implementation, and from [Image2StyleGAN](https://arxiv.org/abs/1904.03189) and [Image2StyleGAN++](https://arxiv.org/abs/1911.11544) papers [unofficial implementations](https://github.com/Jerry2398/Image2StyleGAN-and-Image2StyleGAN-) to generate synthetic images as data augmentation.

The work was carried out by:

- [Domiziano Scarcelli](https://github.com/DomizianoScarcelli)
- [Alessio Lucciola](https://github.com/AlessioLucciola)
- [Danilo Corsi](https://github.com/CorsiDanilo)


## Installation

We use Python 3.10.11 which is the last version supported by PyTorch. To create the environment using conda do

```
conda env create -f environment.yaml
conda activate aml_project
```

## Data

You can download the needed data from this [Google Drive Link](https://drive.google.com/file/d/1vp5x1qXbAubh3p213JC2CwMXYZ7vXdLK/view?usp=drive_link)

Inside the `data` folder, there should be these elements:

-   `HAM10000_images_train`: the directory containing the train images (The original name of the folder is `HAM10000_images_part_1` and `HAM10000_images_part_2`)
-   `HAM10000_segmentations_lesion_tschandl`: the directory containing the masks for the images
-   `HAM10000_metadata_train.csv`: the .csv file containing the metadata for the train images (labels and other info). (The original name of the file is `HAM10000_metadata`)
-   `HAM10000_augmented_images`: The directory containing the GAN-generated images to augment the training set
-   `synthetic_metadata_train.csv`: The .csv file containing the metadata for the GAN generated images.

Moreover, to use SAM, it is necessary to put the `sam_checkpoints.pt` (Download at this [Google Drive Link](https://drive.google.com/file/d/13X_oZo3apJprOS2VTVFND1tfr5TpQJQh/view?usp=drive_link)) file inside the `checkpoints` folder.

## Training
To train a model:
-   Change the configurations you wish to use in the `config.py` file
-   To train the MSLANet model, run `python -m train_loops.MSLANet`. To train a CNN, run `python -m train_loops.CNN_pretrained`. To train a ViT, run `python -m train_loops.ViT`.
-   If you want to resume the training of a model, set `RESUME=True` in the config file, select the folder in which the model is and the checkpoint model. Then start the training again with the command above.

## Testing
To test a model go to `train_loops.test_loop` and select the model to test (checkpoint number or "best"). Then start `python -m train_loops.test_loop`.

## Plots
The plotting function can be found in the `plots` folder. Ensure you have the `tr_val_results.json` and `test_results.json` for a specific model in the `results` folder.
Select the folder name of the model(s) of which to plot the results, then run `python -m plots.test_plots` and `python -m plots.train_plots`.

## Demo
To run the demo, execute `python -m demo.demo_app`. Notice that it is necessary to choose the model to use in the `demo.demo_app` file (the process is the same as the one used in the testing phase).

