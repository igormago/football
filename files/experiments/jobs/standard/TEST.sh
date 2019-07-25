#!/usr/bin/env bash
#SBATCH --partition=TEST
#SBATCH --gres=gpu:1
cd /home/igorcosta/miniconda3/bin
sh ./activate GPU
srun -u python /home/igorcosta/football/models/main.py $*
