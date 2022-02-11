# Vessels Counting and Segmentation

This sample implements the [2018 Data Science Bowl challenge](https://www.kaggle.com/c/data-science-bowl-2018).
The goal is to segment individual vessels in microscopy images.
The `vessel.py` file contains the main parts of the code, and the two Jupyter notebooks


## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)
```
python3 vessel.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model starting from specific weights file using the full `stage1_train` dataset
```
python3 vessel.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
python3 vessel.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Generate submission file from `stage1_test` images
```
python3 vessel.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>
```


## Jupyter notebooks
Two Jupyter notebooks are provided as well: `inspect_vessel_data.ipynb` and `inspect_vessel_model.ipynb`.
They explore the dataset, run stats on it, and go through the detection process step by step.
