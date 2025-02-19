#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen_text
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_colqwen2-v1.3-PairwiseCELoss_text.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_colqwen2-v1.3-PairwiseCELoss_text.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class colqwen2Text \
    --model-name "JFJFJFen/colqwen2-PairwiseCELoss" \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test
