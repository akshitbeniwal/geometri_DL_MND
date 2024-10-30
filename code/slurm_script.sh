#!/bin/bash

#SBATCH --job-name=meshnet_mc
#SBATCH --ntasks=1
#SBATCH --partition=a100                 
#SBATCH --gres=gpu:1                                   
#SBATCH --time=200:00:00

module load cuda
source /home/groups/dlmrimnd/akshit/miniconda3/etc/profile.d/conda.sh
conda activate /home/groups/dlmrimnd/akshit/conda_e/gpu
export PYTHONUNBUFFERED=1
# Run your Python script
# echo "Running PointNet v3"
python r_visual_v2_mc_BA_5c.py