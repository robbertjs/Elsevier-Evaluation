#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=dseqwen2
#SBATCH --mem=30G
#SBATCH --time=15:00:00
#SBATCH --array=1-2
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log2/dseqwen2_arxivqa_screenshot%A_%a.log
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log2/dseqwen2_arxivqa_screenshot%A_%a.log
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

# declare -a IndexSizes=("0.5k" "2.5k" "5k" "7.5k" "10k" "50k")
declare -a IndexSizes=("0.5k" "2.5k" "5k" "7.5k" "10k" "50k")
IndexSize="${IndexSizes[$SLURM_ARRAY_TASK_ID - 1]}"
model_class="dse-qwen2"
data_name="arxivqa"
index_output_name=screenshot_index_${IndexSize}
model_name="MrLight/dse-qwen2-2b-mrl-v1"

#build index 
python /ivi/ilps/personal/jqiao/colpali/vidore-benchmark/src/vidore_benchmark/build_index.py \
    --model-class ${model_class} \
    --model-name ${model_name} \
    --collection-name /ivi/ilps/personal/jqiao/colpali/index_data/${data_name}_test_index_${IndexSize}.jsonl \
    --split test \
    --output-name ${index_output_name}

#search
vidore-benchmark evaluate-retriever \
    --model-class ${model_class} \
    --model-name ${model_name} \
    --collection-name /ivi/ilps/personal/jqiao/colpali/index_data/${data_name}_test_index_${IndexSize}.jsonl \
    --split test \
    --indexing-path /ivi/ilps/personal/jqiao/colpali/outputs/indexing/${model_class}_${data_name}_indexing_results_${index_output_name}.pt \
    --data-index-name ${index_output_name}_${data_name}
