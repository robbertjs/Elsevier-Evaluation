#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=colpali
#SBATCH --mem=30G
#SBATCH --array=1-4
#SBATCH --time=80:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log3/rq3_colpali_text%A_%a.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log3/rq3_colpali_text%A_%a.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task


declare -a IndexSizes=("tabfquad_test_subsampled_tesseract" "shiftproject_test_tesseract" "arxivqa_test_subsampled_tesseract" "syntheticDocQA_healthcare_industry_test_tesseract")
data_name="${IndexSizes[$SLURM_ARRAY_TASK_ID - 1]}"
model_class="colpali2Text"
model_name="JFJFJFen/colpali-PairwiseCELoss"


vidore-benchmark evaluate-retriever \
    --model-class ${model_class} \
    --model-name "${model_name}" \
    --dataset-name "vidore/${data_name}" \
    --split test \
    --matching-type text_lexical

vidore-benchmark evaluate-retriever \
    --model-class ${model_class} \
    --model-name "${model_name}" \
    --dataset-name "vidore/${data_name}" \
    --split test \
    --matching-type text_nonlexical

vidore-benchmark evaluate-retriever \
    --model-class ${model_class} \
    --model-name "${model_name}" \
    --dataset-name "vidore/${data_name}" \
    --split test \
    --matching-type text_special_token

vidore-benchmark evaluate-retriever \
    --model-class ${model_class} \
    --model-name "${model_name}" \
    --dataset-name "vidore/${data_name}" \
    --split test \
    --matching-type text_qtm






