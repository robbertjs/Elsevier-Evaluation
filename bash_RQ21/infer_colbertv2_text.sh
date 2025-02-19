#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=jina
#SBATCH --mem=30G
#SBATCH --time=80:00:00
#SBATCH --array=1
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/infer_colbert.log
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/infer_colbert.log
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   

vidore-benchmark evaluate-retriever \
    --model-class jina-colbert \
    --model-name colbert-ir/colbertv2.0 \
    --collection-name "vidore/vidore-page-ocr-artifact-6669de61f09324d7d940dd53" \
    --split test