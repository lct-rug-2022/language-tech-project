# Task 2: Analysing ChangeMyView Discussions

Data downloaded from https://zenodo.org/record/3778298#.ZFpSu3ZByUk

## Data Pre-Processing:
* Extract `threads.jsonl.bz2` in `discussion-task/data/` folder
* Run `python discussion-task/data/process.py`

## Inference:
* Run `python discussion-task/predict.py`

## ChangeMyView Analysis:
* Run `python discussion-task/hv_dist.py`
* Run `python discussion-task/alignment_correlation.py`