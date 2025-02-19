#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=gte_qwen2-7b
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/gte_qwen2-7b.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/gte_qwen2-7b.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:4   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class "gte" \
    --model-name Alibaba-NLP/gte-Qwen2-7B-instruct \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test