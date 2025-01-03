#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colpali
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/colpali.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/colpali.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request one GPU per task

accelerate launch /ivi/ilps/personal/jqiao/colpali/scripts/train/train_colbert.py \
    /ivi/ilps/personal/jqiao/colpali/scripts/configs/pali/train_colpali2_pt_model.yaml
