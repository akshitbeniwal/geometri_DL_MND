import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os

# List of subject/session identifiers
subjects_sessions = [
    'sub-002MND/ses-01', 'sub-002MND/ses-02', 'sub-003MND/ses-03'
]

base_input_dir = "../../data/combined-bids"
base_output_dir = "../../data/combined-bids"

# Function to process each file
def process_file(subject_session):
    input_file = os.path.join(base_input_dir, subject_session, 'anat', f"{subject_session.replace('/', '_')}_T1w.nii.gz")
    output_file = os.path.join(base_output_dir, subject_session, 'anat', f"{subject_session.replace('/', '_')}_T1w.nii.gz")

    # Load the original MRI brain scan image
    img = nib.load(input_file)
    data = img.get_fdata()

    # Remove the neck area by slicing the third dimension from index 60 onwards
    trimmed_data = data[:, :, 60:]

    # Calculate the zoom factors to resize the image to 256 x 256 x 256
    zoom_factors = [256 / s for s in trimmed_data.shape]

    # Resize the image
    resized_data = zoom(trimmed_data, zoom_factors, order=1)  # Use order=1 for linear interpolation

    # Save the new image
    resized_img = nib.Nifti1Image(resized_data, img.affine)
    nib.save(resized_img, output_file)

    print(f"Processed {subject_session} successfully.")

# Process each file in the list
for subject_session in subjects_sessions:
    process_file(subject_session)

