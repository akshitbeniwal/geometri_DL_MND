#!/bin/bash

#SBATCH --job-name=pointnet
#SBATCH --ntasks=1 
#SBATCH --partition=a100                 
#SBATCH --gres=gpu:1                                    
#SBATCH --time=24:00:00

module load cuda
source /home/groups/dlmrimnd/akshit/miniconda3/etc/profile.d/conda.sh
conda activate /home/groups/dlmrimnd/akshit/conda_e/gpu

# Run your Python script
python pointnet.py