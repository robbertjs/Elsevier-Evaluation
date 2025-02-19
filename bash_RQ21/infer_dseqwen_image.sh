#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=dse
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log4/inference_dse.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log4/inference_dse.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


vidore-benchmark evaluate-retriever \
    --model-class dse-qwen2 \
    --model-name MrLight/dse-qwen2-2b-mrl-v1 \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test
