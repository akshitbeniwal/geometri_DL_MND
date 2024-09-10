#!/bin/bash

#SBATCH --job-name=fastsurfer_job1 
#SBATCH --ntasks=1 
#SBATCH --partition=p100                 
#SBATCH --gres=gpu:1                                    
#SBATCH --time=24:00:00 

# Set paths based on your arguments
outputdir="$1"
datadir="$2"
# fastsurfer_image="$3"  # Use the provided argument
sid="$3"
# license="$5"

data_dir="/home/groups/dlmrimnd/akshit/data:/data" 
output_dir="/home/groups/dlmrimnd/akshit/output_files:/MND_output_files"
FS_dir="/home/groups/dlmrimnd/akshit/freesurfer_license:/freesurfer_license"
# Load the Singularity module (if needed)


# Singularity execution command
# singularity run --nv --no-home \
#   -B "$data_dir" \
#   -B "$output_dir" \
#   -B "$FS_dir" \
#   "$fastsurfer_image" \
#   --fs_license "$license" \
#   --sd "$outputdir" \
#   --t1 "$datadir" \
#   --sid "$sid"
export TMPDIR="../../temp"
singularity exec --nv --no-home \
                ../fastsurfer-latest.sif \
                ./run_fastsurfer.sh \
                --fs_license ../../freesurfer_license/license.txt \
                --t1 "$datadir" \
                --sid "$sid" \
                --sd "$outputdir" \
                --py python --fsqsphere