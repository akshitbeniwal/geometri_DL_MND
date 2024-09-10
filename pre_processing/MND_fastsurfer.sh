#!/bin/bash

# Path to the CSV file
csv_file="MND_patient_filepaths.csv"

# Output directory
output_dir="../../MND_output_files/"

# Read the CSV file line by line
while IFS=, read -r filepath filename
do
    # Skip the header line
    if [ "$filepath" != "filepath" ]; then
        # Remove the .nii.gz extension from the filename to get the subject_id
        subject_id="${filename%.nii.gz}"
        
        # Construct the expected directory path
        subject_dir="$output_dir/$subject_id/surf"
        
        # Check if the directory exists
        # if [ -d "$subject_dir" ]; then
        #     # Check if the file exists
        #     if [ -f "$subject_dir/lh.qsphere.nofix" ]; then
                # Submit the sbatch job
        sbatch srun_fastsurfer_user.sh "$output_dir" "$filepath" "$subject_id"
        #     else
        #         echo "File $subject_dir/lh.qsphere.nofix does not exist."
        #     fi
        # else
        #     echo "Directory $subject_dir does not exist."
        # fi
    fi
done < "$csv_file"
