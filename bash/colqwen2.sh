#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/colqwen.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/colqwen.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request one GPU per task

accelerate launch --main_process_port 29501 /ivi/ilps/personal/jqiao/colpali/scripts/train/train_colbert.py \
    /ivi/ilps/personal/jqiao/colpali/scripts/configs/qwen2/train_colqwen2_model.yaml