#!/bin/sh
#SBATCH -p gpu
#SBATCH --job-name=coverage
#SBATCH --mem=30G
#SBATCH --time=70:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/coverage.output
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/coverage.output
#SBATCH --gres=gpu:nvidia_rtx_a6000:1   # Request one GPU per task

datasets=(arxivqa_test_subsampled docvqa_test_subsampled infovqa_test_subsampled tabfquad_test_subsampled tatdqa_test shiftproject_test syntheticDocQA_artificial_intelligence_test syntheticDocQA_energy_test syntheticDocQA_government_reports_test syntheticDocQA_healthcare_industry_test)

for data_name in "${datasets[@]}"
do
    echo "Processing dataset: $data_name"
    CUDA_VISIBLE_DEVICES=0 python /ivi/ilps/personal/jqiao/colpali/scripts/coverage.py \
        --dataset "vidore/$data_name" \
        --output "/ivi/ilps/personal/jqiao/colpali/outputs/converage2/${data_name}.jsonl"
done