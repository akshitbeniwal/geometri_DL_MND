#!/bin/bash

# Path to the directory containing your CSV file
csv_dir="./data_sum/"

# Full path to the directory containing your data
data_dir="../../data/ADNI_HIPPO_T1w_ONLY/"

# Full path to the directory where you want to store output
output_dir="../../output_files/"

# Path to the fastsurfer Singularity image
fastsurfer_image="../fastsurfer-latest.sif"

# Loop through each line in the CSV file
while IFS= read -r filename; do
    # Remove any carriage return characters from the filename
    filename=$(echo "$filename" | tr -d '\r')

    # Construct the full path to the subject folder
    subject_dir="${data_dir}${filename}/"

    # Debug: Print the subject directory
    echo "Subject Directory: $subject_dir"

    # Loop through each session folder within the subject folder
    # Loop through each session folder within the subject folder
while IFS= read -r -d '' session_dir; do
    echo "${session_dir} hello"
    # Check if the session folder exists and contains the required file
    if [ -f "${session_dir}/anat/${filename}_ses-${session_dir##*-}_T1w.nii.gz" ]; then
        # Construct the full path to the file
        file_path="${session_dir}/anat/${filename}_ses-${session_dir##*-}_T1w.nii.gz"
        echo "$file_path"
        # Extract the subject ID from the filename
        subject_id=$(basename "${filename}")
        
        # Print the variables
        SBATCH srun_fastsurfer_user.sh "$output_dir" "$file_path" "$subject_id"
    fi
    done < <(find "${subject_dir}" -mindepth 2 -maxdepth 2 -type d -name "ses-*" -print0)
done < "${csv_dir}ADNI_NC_filenames.csv"
