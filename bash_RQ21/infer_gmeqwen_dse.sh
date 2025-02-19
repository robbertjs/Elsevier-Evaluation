#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen2
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/gme-Qwen2-VL-2B-Instruct_dse.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/gme-Qwen2-VL-2B-Instruct_dse.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class "gme-qwen2" \
    --model-name Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test
