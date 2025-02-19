#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=infer
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --array=1
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/create_index_arxivqa.log
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/create_index_arxivqa.log
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


python /ivi/ilps/personal/jqiao/colpali/scripts/create_index_arxivqa.py