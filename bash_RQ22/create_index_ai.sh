#!/bin/bash
#SBATCH --job-name=index
#SBATCH --mem=30G
#SBATCH --time=50:00:00
#SBATCH --output=/ivi/ilps/personal/jqiao/colpali/log/create_index_ai.log
#SBATCH --error=/ivi/ilps/personal/jqiao/colpali/log/create_index_ai.log
#SBATCH --array=1
#SBATCH --partition=cpu

python /ivi/ilps/personal/jqiao/colpali/scripts/create_index_ai.py
