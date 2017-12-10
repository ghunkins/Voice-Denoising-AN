#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00:00
#SBATCH --mem 64GB
#SBATCH --job-name=pix
#SBATCH --output=output_pix_%j.txt
#SBATCH -e error_pix_%j.txt
#SBATCH --gres=gpu:2

source activate new_pix
python main.py --backend tensorflow --dset audio_1000 --nb_epoch 20 64 64