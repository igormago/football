#!/usr/bin/env bash
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
cd /home/igorcosta/miniconda3/bin
sh ./activate CPU
srun -u python /home/igorcosta/football/models/main.py $*
