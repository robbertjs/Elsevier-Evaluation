#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen2
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/dse-qwen2-text.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/dse-qwen2-text.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class dse-qwen2-text \
    --model-name MrLight/dse-qwen2-2b-mrl-v1 \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test

