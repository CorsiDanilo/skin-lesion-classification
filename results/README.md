This folder is used to save models and training/Testing results. In particular:
- Models are saved if config.SAVE_MODELS=True
- Results (in json) are saved if config.SAVE_RESULTS=True

Once the training is started, a new folder named "model_starting-datetime" will be created.
This folder has a configurations.json file with the settings used for training the model.
There are two subfolders:
- results: It contains tr_val_results.json with the training/validation results for each epoch and test_results.json with the results obtained during testing.
- models: It contains the checkpoints created at each epoch.