#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=indexing
#SBATCH --mem=120G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/build_index.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/build_index.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

python /ivi/ilps/personal/jqiao/colpali/vidore-benchmark/src/vidore_benchmark/build_index.py \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test

