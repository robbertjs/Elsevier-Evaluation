#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=bipali
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/reproduce_bilpali-v1.3-PairwiseCELoss2.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/reproduce_bilpali-v1.3-PairwiseCELoss2.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request one GPU per task


accelerate launch --main_process_port 29502 /ivi/ilps/personal/jqiao/colpali/scripts/train/train_colbert.py \
    /ivi/ilps/personal/jqiao/colpali/scripts/configs/pali/reproduce_bipali-v1.3-PairwiseCELoss.yaml