#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen2
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_colqwen2_repro.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_colqwen2_repro.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class colqwen2 \
    --model-name "JFJFJFen/colqwen2-PairwiseCELoss" \
    --collection-name "vidore/vidore-benchmark-667173f98e70a1c0fa4db00d" \
    --split test
