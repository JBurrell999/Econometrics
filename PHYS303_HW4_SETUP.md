# PHYS 303 HW4 Setup

This repo already contains a working cats-vs-dogs homework solution in `phys303_hw4_cats_dogs.py` and `phys303_hw4_cats_dogs.ipynb`.

The code is preconfigured for the local Kaggle dataset path:

`/Users/jjburrell/Downloads/kagglecatsanddogs_5340/PetImages`

## Install dependencies

```bash
python3 -m pip install -r requirements-phys303-hw4.txt
```

## Run the homework script

```bash
python3 phys303_hw4_cats_dogs.py
```

Useful options:

```bash
python3 phys303_hw4_cats_dogs.py --device auto
python3 phys303_hw4_cats_dogs.py --epochs 4
python3 phys303_hw4_cats_dogs.py --batch-size 64
python3 phys303_hw4_cats_dogs.py --output-dir phys303_hw4_outputs
```

## What the code does

- Scans `Cat/` and `Dog/` folders.
- Removes unreadable or zero-byte files and records them in `removed_images.csv`.
- Converts every image to RGB during loading so mixed image modes are handled safely.
- Uses an 80/20 train-validation split.
- Trains a small CNN for binary classification.
- Saves:
  - `best_model.pt`
  - `metrics.json`
  - `loss_vs_epoch.png`
  - `accuracy_vs_epoch.png`
  - `roc_curve.png`
  - `precision_recall_curve.png`
  - `removed_images.csv`

## Existing outputs already in the repo

The repo already includes completed output folders from earlier runs, including `phys303_hw4_outputs` and `phys303_hw4_outputs_notebook`.

From the saved metrics:

- Valid images: `24,998`
- Removed/corrupted images: `2`
- Validation accuracy: `0.7204`
- Validation ROC AUC: `0.8095`
- Validation average precision: `0.8051`

## Notebook option

If you want the homework in notebook form, open:

`phys303_hw4_cats_dogs.ipynb`

That notebook follows the same PHYS-style workflow:

- preprocessing
- dataset and dataloader setup
- model definition
- training loop
- saved plots and metrics
