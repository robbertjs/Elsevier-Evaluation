#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colqwen_text
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_biqwen2-zeroshot_text.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_biqwen2-zeroshot_text.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

vidore-benchmark evaluate-retriever \
    --model-class dse-qwen2-text \
    --model-name "Qwen/Qwen2-VL-2B-Instruct" \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test