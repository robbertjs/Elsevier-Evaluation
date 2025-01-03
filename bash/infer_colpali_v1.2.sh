#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colpali
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/inference2.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/inference2.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


vidore-benchmark evaluate-retriever \
    --model-class colpali \
    --model-name vidore/colpali-v1.2 \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test
