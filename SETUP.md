## Training

### Colab training 

Just copy specific script folder content to `/content/`: 
```shell
content/
  - params.json
  - train.py
  - requirements.txt
```

Export `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`, login into `HF Hub` via `huggingface_login`
```shell
%env NEPTUNE_PROJECT=...
%env NEPTUNE_API_TOKEN=...
```

And install required libs
```shell
!pip install -U -r requirements.txt
```


### Habrok Run


#### Copy files (each time)

0. Copy files to the server, e.g:
   ```shell
   # if you have "habrok" as alias for the username@login1.hb.....rug.nl
   scp -r ./ habrok:~/language-tech-project  
   ```
   Or setup PyCharm deployment (settings -> deployment -> add new -> SFTP)

#### Habrok setup (once)

ALL INSIDE REPO 

1. Create `.env` file with `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`
2. Load modules same as in `jobscript` (for baseline it is `single_task.sh`)
   ```shell
   module purge
   module load CUDA/11.7.0
   module load cuDNN/8.4.1.50-CUDA-11.7.0
   module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
   module load Python/3.10.4-GCCcore-11.3.0
   module load GCC/11.3.0
   ```
3. Create new venv and make `.venv` alias in the folder
    ```shell
    python3 -m venv /home/$USER/.envs/language-tech-project
    ln -d -s /home/$USER/.envs/language-tech-project .venv
    ```
4. Install libs 
    ```shell
    source .venv/bin/activate
    pip install -U -r requirements.txt
    ```
5. Make `results` folder at `/scratch` to log there (more space)
    ```shell
    mkdir -p /scratch/$USER/language-tech-project/results
    ln -d -s /scratch/$USER/language-tech-project/results results
6. Make `models` folder at `/scratch` to log there (more space)
    ```shell
    mkdir -p /scratch/$USER/language-tech-project/models
    ln -d -s /scratch/$USER/language-tech-project/models models
    ```
   
#### Each session setup 

Ether run sbatch jobs only or setup modules (see above) and activate venv. 
   
#### Habrok run

0. Copy files to the server, e.g:
   ```shell
   # if you have "habrok" as alias for the username@login1.hb.....rug.nl
   scp -r ./ habrok:~/language-tech-project  
   ```
   Or setup PyCharm deployment (settings -> deployment -> add new -> SFTP)
1. Run job script 
    ```shell
    sbatch scripts/values-task/train.sh --base-model=roberta-base [other options]
    ```
2. Monitor status with 
    ```shell
    squeue | grep $USER
    squeue | grep gpu
    squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.18R %p" | grep gpu
    ```


Example of running job with values-task finetuning on roberta-base:
```shell
sbatch scripts/values-task/train.sh --base-model=roberta-base --config-name=default
```
