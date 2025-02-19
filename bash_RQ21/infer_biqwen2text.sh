#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=biqwen2text
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_biqwen2text-v1.3-PairwiseCELoss.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_biqwen2text-v1.3-PairwiseCELoss.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


vidore-benchmark evaluate-retriever \
    --model-class biqwen2text \
    --model-name biqwen2-PairwiseCELoss \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test
