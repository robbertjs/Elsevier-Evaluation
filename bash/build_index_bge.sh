#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=indexing
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/build_index_bge.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/build_index_bge.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

python /ivi/ilps/personal/jqiao/colpali/vidore-benchmark/src/vidore_benchmark/build_index.py \
    --model-class bge-m3-colbert \
    --model-name BAAI/bge-m3 \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test
