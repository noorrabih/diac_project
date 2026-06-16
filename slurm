#! /bin/bash
#SBATCH --job-name=full_pipeline # Job name
#SBATCH --output=full_pipeline_train.txt # Standard output and error.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos
python barec10M_words_diac_analytics.py 