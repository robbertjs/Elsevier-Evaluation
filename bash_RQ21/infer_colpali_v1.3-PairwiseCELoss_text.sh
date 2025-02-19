#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colpali
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_colpali-v1.3-PairwiseCELoss_text.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_colpali-v1.3-PairwiseCELoss_text.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


vidore-benchmark evaluate-retriever \
    --model-class colpali2Text \
    --model-name "JFJFJFen/colpali-PairwiseCELoss" \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test
