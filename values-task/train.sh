#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --job-name=values-task
#SBATCH --mem=20G


module purge
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load GCC/11.3.0


source .venv/bin/activate


export $(cat .env | xargs)
export NEPTUNE_PROJECT="lct-rug-2022/language-tech-project"
export TOKENIZERS_PARALLELISM=false


python values-task/train.py $*


deactivate
