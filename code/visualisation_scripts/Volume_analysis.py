import os
import re
import csv
import matplotlib.pyplot as plt

def extract_data(file_path):
    # Initialize variables to store extracted values
    paracentral_vol = None
    precentral_vol = None
    etiv = None

    # Read the file and extract the needed values
    with open(file_path, 'r') as f:
        for line in f:
            if 'paracentral' in line:
                paracentral_vol = float(line.split()[3])
            elif 'precentral' in line:
                precentral_vol = float(line.split()[3])
            elif 'EstimatedTotalIntraCranialVol' in line:
                etiv = float(line.split()[8].strip(','))

    return paracentral_vol, precentral_vol, etiv

def plot_sorted_data(patient_ids, normalized_values, title, ylabel, dest_folder, filename):
    # Sort data
    sorted_data = sorted(zip(patient_ids, normalized_values), key=lambda x: x[1])
    sorted_patient_ids, sorted_normalized_values = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(20, 12))
    plt.plot(sorted_patient_ids, sorted_normalized_values, marker='o')
    plt.xlabel('Patient ID')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(dest_folder, filename)
    plt.savefig(plot_path)
    plt.show()

def process_files(csv_file, src_root, dest_folder):
    data = {}

    # Open the CSV file and read the folder names and labels
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            folder_path = os.path.join(src_root, folder)  # Construct the folder path

            stats_dir = os.path.join(folder_path, 'stats')
            mri_dir = os.path.join(folder_path, 'mri')

            lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
            rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

            if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
                lh_paracentral_vol, lh_precentral_vol, lh_etiv = extract_data(lh_stats_file)
                rh_paracentral_vol, rh_precentral_vol, rh_etiv = extract_data(rh_stats_file)

                # Sum volumes from left and right hemispheres
                paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
                precentral_vol = lh_precentral_vol + rh_precentral_vol
                etiv = lh_etiv  # eTIV should be the same in both files, use either

                data[folder] = {
                    'paracentral': paracentral_vol,
                    'precentral': precentral_vol,
                    'etiv': etiv
                }

    # Prepare data for plotting
    patient_ids = list(data.keys())
    normalized_paracentral = [data[pid]['paracentral'] / data[pid]['etiv'] for pid in patient_ids]
    normalized_precentral = [data[pid]['precentral'] / data[pid]['etiv'] for pid in patient_ids]

    # Plot sorted data separately for paracentral and precentral
    plot_sorted_data(patient_ids, normalized_paracentral,
                     title='Normalized Paracentral Gyrus Volume (Sorted)',
                     ylabel='Normalized Paracentral Volume',
                     dest_folder=dest_folder,
                     filename='sorted_paracentral_volumes.png')

    plot_sorted_data(patient_ids, normalized_precentral,
                     title='Normalized Precentral Gyrus Volume (Sorted)',
                     ylabel='Normalized Precentral Volume',
                     dest_folder=dest_folder,
                     filename='sorted_precentral_volumes.png')

# Define source and destination paths
csv_file = '/path/to/your/csv_file.csv'  # Path to your CSV file
src_root = '/MND_patients'
dest_folder = '/path/to/output'

# Make sure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Run the processing function
process_files(csv_file, src_root, dest_folder)
