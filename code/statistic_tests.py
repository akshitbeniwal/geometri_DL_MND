import os
import re
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations

def extract_data(file_path):
    # Initialize variables to store extracted values
    paracentral_vol = None
    precentral_vol = None
    etiv = None
    cortical_thickness = None

    # Read the file and extract the needed values
    with open(file_path, 'r') as f:
        for line in f:
            if 'paracentral' in line:
                paracentral_vol = float(line.split()[3])
            elif 'precentral' in line:
                precentral_vol = float(line.split()[3])
            elif 'EstimatedTotalIntraCranialVol' in line:
                etiv = float(line.split()[8].strip(','))
            elif 'Measure Cortex, MeanThickness' in line:
                cortical_thickness = float(line.split()[6].strip(','))

    return paracentral_vol, precentral_vol, etiv, cortical_thickness

def plot_sorted_data(patient_ids, normalized_values, title, ylabel, dest_folder, filename):
    # Sort data
    sorted_data = sorted(zip(patient_ids, normalized_values), key=lambda x: x[1])
    sorted_patient_ids, sorted_normalized_values = zip(*sorted_data)

    # Plotting
    plt.figure(figsize=(24, 16))
    plt.plot(sorted_patient_ids, sorted_normalized_values, marker='o')
    plt.xlabel('Patient ID')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(dest_folder, filename)
    plt.savefig(plot_path)
    plt.show()

def process_files(csv_file, src_root, dest_folder):
    data = []

    # Open the CSV file and read the folder names and labels
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            diagnosis = int(row['formal.diagnosis.numeric'])
            folder_path = os.path.join(src_root, folder)

            stats_dir = os.path.join(folder_path, 'stats')

            lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
            rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

            if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
                lh_paracentral_vol, lh_precentral_vol, lh_etiv, lh_cortical_thickness = extract_data(lh_stats_file)
                rh_paracentral_vol, rh_precentral_vol, rh_etiv, rh_cortical_thickness = extract_data(rh_stats_file)

                # Sum volumes from left and right hemispheres
                paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
                precentral_vol = lh_precentral_vol + rh_precentral_vol
                etiv = lh_etiv  # eTIV should be the same in both files, use either
                cortical_thickness = lh_cortical_thickness  # Cortical thickness should be the same, use either

                data.append({
                    'folder': folder,
                    'diagnosis': diagnosis,
                    'paracentral': paracentral_vol / etiv,
                    'precentral': precentral_vol / etiv,
                    'cortical_thickness': cortical_thickness
                })

    return pd.DataFrame(data)

def run_anova_and_ttests(df, metric):
    # Run ANOVA
    formula = f'{metric} ~ C(diagnosis)'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f'ANOVA for {metric}:\n', anova_table)

    # If ANOVA is significant (p < 0.05), do pairwise t-tests
    if anova_table['PR(>F)'][0] < 0.05:
        print(f'\nSignificant ANOVA for {metric}. Running pairwise t-tests...\n')
        diagnoses = df['diagnosis'].unique()
        pairs = list(combinations(diagnoses, 2))

        for (d1, d2) in pairs:
            group1 = df[df['diagnosis'] == d1][metric]
            group2 = df[df['diagnosis'] == d2][metric]
            t_stat, p_val = ttest_ind(group1, group2)
            print(f'T-test between {d1} and {d2} for {metric}: t-stat = {t_stat:.4f}, p-val = {p_val:.4f}')

def main(csv_file, src_root, dest_folder):
    # Process files and get data
    df = process_files(csv_file, src_root, dest_folder)

    # Perform ANOVA and pairwise t-tests for cortical thickness and paracentral volume
    run_anova_and_ttests(df, 'cortical_thickness')
    run_anova_and_ttests(df, 'paracentral')
    run_anova_and_ttests(df, 'precentral')
    # Plot sorted data
    # plot_sorted_data(df['folder'], df['paracentral'], 
    #                  title='Normalized Paracentral Gyrus Volume (Sorted)', 
    #                  ylabel='Normalized Paracentral Volume', 
    #                  dest_folder=dest_folder, 
    #                  filename='MND_sorted_paracentral_volumes.png')

    # plot_sorted_data(df['folder'], df['cortical_thickness'], 
    #                  title='Cortical Mean Thickness (Sorted)', 
    #                  ylabel='Mean Cortical Thickness (mm)', 
    #                  dest_folder=dest_folder, 
    #                  filename='MND_sorted_cortical_thickness.png')

# Define source and destination paths
csv_file = '/home/groups/dlmrimnd/akshit/New_MND_patient_data_summary.csv'  # Path to your CSV file
src_root = '/home/groups/dlmrimnd/akshit/MND_patients/'
dest_folder = '/home/groups/dlmrimnd/akshit/geometricDL_MND/images'

# Make sure the destination folder exists
os.makedirs(dest_folder, exist_ok=True)

# Run the main function
main(csv_file, src_root, dest_folder)
