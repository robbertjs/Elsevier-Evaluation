#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=bipali
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_bipali-v1.3-PairwiseCELoss2.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_bipali-v1.3-PairwiseCELoss2.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class bipali \
    --model-name "JFJFJFen/bipali-PairwiseCELoss" \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test