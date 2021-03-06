#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00:00
#SBATCH --mem 10GB
#SBATCH --job-name=pix_val
#SBATCH --output=o_validation_%j.txt
#SBATCH -e e_validation_%j.txt
#SBATCH --gres=gpu:2

source activate new_pix
python validation.py 