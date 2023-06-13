# Task 1: Human Values Detection

Data downloaded from https://zenodo.org/record/7879430

## Training:
```shell
sbatch ./train.sh --config-name=[config-name] --base-model=[base-model] --push-to-hub --aug-type=[aug-type] --save-model
```
Training script accept the following arguments:
* `--base-model` - model to finetune available at Hugginface Hub (e.g. `roberta-base`)
* `--config-name` - name of the config in `params.json` file (separate for each script)
* `--aug_type` - augmentation type to apply
* `--push-to-hub` - pushing trained model to HuggingFace Hub
* `--save-model` - save trained model locally

See more `python [script_name].py --help`