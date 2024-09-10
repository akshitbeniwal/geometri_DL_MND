

import nibabel as nib
from nibabel.processing import conform
import numpy as np
import os

# List of subject identifiers
subjects = [
    'sub-041S6136_ses_02', 'sub-041S6354_ses_01', 'sub-024S6385_ses_01',
    'sub-127S6348_ses_01', 'sub-016S6381_ses_02', 'sub-099S6016_ses_01',
    'sub-041S6292_ses_01', 'sub-301S6224_ses_01', 'sub-041S6136_ses_01',
    'sub-041S6354_ses_02', 'sub-941S6094_ses_01', 'sub-941S6054_ses_01',
    'sub-016S6381_ses_01', 'sub-041S6292_ses_02', 'sub-041S6192_ses_02',
    'sub-041S6192_ses_01', 'sub-024S6005_ses_01', 'sub-024S6202_ses_01'
]

# Base input and output directories
input_base_dir = '/home/groups/dlmrimnd/akshit/data/ADNI_HIPPO_T1w_ONLY'
output_base_dir = '/home/groups/dlmrimnd/akshit/Input_ADNI_fault_files'

# Loop through each subject
for subject in subjects:
    parts = subject.split('_')
    ZZZ = parts[0]
    YYY = parts[1]
    XXX = subject
    
    input_file = os.path.join(input_base_dir, ZZZ, YYY, 'anat', f'{XXX}_T1w.nii.gz')
    output_file = os.path.join(output_base_dir, f'{XXX}_T1w.nii.gz')
    
    # Load the image
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Average the data along the time axis (axis=3)
    if data.ndim == 4:  # Check if data has 4 dimensions
        average_data = np.mean(data, axis=3)
    else:
        average_data = data  # No need to average if data is already 3D
    
    # Create a new NIfTI image with the averaged data
    average_img = nib.Nifti1Image(average_data, img.affine)
    
    # Conform the image
    conformed_img = conform(average_img, out_shape=(256, 256, 256), voxel_size=(1, 1, 1), order=1)
    
    # Save the conformed image
    nib.save(conformed_img, output_file)
    
    print(f'Processed {subject}')
