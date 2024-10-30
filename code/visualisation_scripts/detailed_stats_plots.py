# import os
# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import f_oneway, ttest_ind, pearsonr
# import seaborn as sns
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# # Define diagnosis mapping
# DIAGNOSIS_MAPPING = {
#     0: 'Normal Control',
#     1: 'ALS',
#     2: 'PLS',
#     4: 'PMA'
#     # Add more mappings as necessary
# }

# # Define custom color palette
# CUSTOM_COLORS = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']

# def extract_data(file_path):
#     paracentral_vol = None
#     precentral_vol = None
#     etiv = None
#     cortical_thickness = None

#     with open(file_path, 'r') as f:
#         for line in f:
#             if 'paracentral' in line:
#                 paracentral_vol = float(line.split()[3])
#             elif 'precentral' in line:
#                 precentral_vol = float(line.split()[3])
#             elif 'EstimatedTotalIntraCranialVol' in line:
#                 etiv = float(line.split()[8].strip(','))
#             elif 'Measure Cortex, MeanThickness' in line:
#                 cortical_thickness = float(line.split()[6].strip(','))

#     return paracentral_vol, precentral_vol, etiv, cortical_thickness

# def process_mnd_files(csv_file, src_root_mnd):
#     data = []

#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             folder = row['folder']
#             if folder == str(108):
#                 continue  # Skip folder 108 as per user request
#             diagnosis_num = row.get('formal.diagnosis.numeric', row.get('label', '0'))
#             try:
#                 diagnosis_num = int(diagnosis_num)
#             except ValueError:
#                 diagnosis_num = 0  # Default to control if invalid

#             diagnosis = DIAGNOSIS_MAPPING.get(diagnosis_num, 'Other')

#             folder_path = os.path.join(src_root_mnd, folder)

#             stats_dir = os.path.join(folder_path, 'stats')
#             lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
#             rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

#             if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
#                 lh_paracentral_vol, lh_precentral_vol, lh_etiv, lh_cortical_thickness = extract_data(lh_stats_file)
#                 rh_paracentral_vol, rh_precentral_vol, rh_etiv, rh_cortical_thickness = extract_data(rh_stats_file)

#                 if lh_etiv and rh_etiv:
#                     etiv = lh_etiv  # Assuming ETIV is the same for both hemispheres
#                     paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
#                     precentral_vol = lh_precentral_vol + rh_precentral_vol
#                     cortical_thickness = (lh_cortical_thickness + rh_cortical_thickness) / 2  # Average

#                     months_since_onset = row.get('months.since.onset', np.nan)
#                     try:
#                         months_since_onset = float(months_since_onset)
#                     except ValueError:
#                         months_since_onset = np.nan

#                     data.append({
#                         'folder': folder,
#                         'diagnosis_num': diagnosis_num,
#                         'diagnosis': diagnosis,
#                         'paracentral': paracentral_vol / etiv,
#                         'precentral': precentral_vol / etiv,
#                         'cortical_thickness': cortical_thickness,
#                         'age_at_scan': row.get('age.at.scan', np.nan),
#                         'sex': row.get('sex.numerical', np.nan),
#                         'months_since_onset': months_since_onset
#                     })

#     return pd.DataFrame(data)

# def process_control_files(csv_file, src_root_control):
#     control_data = []

#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             folder = row['folder']
#             diagnosis_num = int(row.get('label', 0))
#             diagnosis = DIAGNOSIS_MAPPING.get(diagnosis_num, 'Normal Control')  # Assuming label=0 is control

#             folder_path = os.path.join(src_root_control, folder)

#             stats_dir = os.path.join(folder_path, 'stats')
#             lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
#             rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

#             if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
#                 lh_paracentral_vol, lh_precentral_vol, lh_etiv, lh_cortical_thickness = extract_data(lh_stats_file)
#                 rh_paracentral_vol, rh_precentral_vol, rh_etiv, rh_cortical_thickness = extract_data(rh_stats_file)

#                 if lh_etiv and rh_etiv:
#                     etiv = lh_etiv  # Assuming ETIV is the same for both hemispheres
#                     paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
#                     precentral_vol = lh_precentral_vol + rh_precentral_vol
#                     cortical_thickness = (lh_cortical_thickness + rh_cortical_thickness) / 2  # Average

#                     age_at_scan = row.get('age.at.scan', np.nan)
#                     try:
#                         age_at_scan = float(age_at_scan)
#                     except ValueError:
#                         age_at_scan = np.nan

#                     sex = row.get('sex.numerical', np.nan)
#                     try:
#                         sex = int(sex)
#                     except ValueError:
#                         sex = np.nan

#                     control_data.append({
#                         'folder': folder,
#                         'diagnosis_num': diagnosis_num,
#                         'diagnosis': diagnosis,
#                         'paracentral': paracentral_vol / etiv,
#                         'precentral': precentral_vol / etiv,
#                         'cortical_thickness': cortical_thickness,
#                         'age_at_scan': age_at_scan,
#                         'sex': sex,
#                         'months_since_onset': np.nan  # Not applicable for controls
#                     })

#     return pd.DataFrame(control_data)
# from scipy.stats import levene
# from statsmodels.stats.weightstats import ttest_ind as welch_ttest

# def run_anova_and_ttests(df, metric):
#     formula = f'{metric} ~ C(diagnosis)'
#     model = ols(formula, data=df).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(f'\nANOVA for {metric}:\n', anova_table)

#     print(f'\nTesting for equal variances and performing pairwise t-tests for {metric}:')
#     diagnoses = sorted(df['diagnosis'].unique())
#     variance_results = {}
    
#     # Perform Levene’s test to check for equal variances
#     for i in range(len(diagnoses)):
#         for j in range(i + 1, len(diagnoses)):
#             d1, d2 = diagnoses[i], diagnoses[j]
#             group1 = df[df['diagnosis'] == d1][metric].dropna()
#             group2 = df[df['diagnosis'] == d2][metric].dropna()
            
#             if len(group1) > 1 and len(group2) > 1:
#                 # Levene's test
#                 stat, p = levene(group1, group2)
#                 variance_results[(d1, d2)] = p
#                 print(f'Levene’s test between {d1} and {d2}: p-value = {p:.4f}')
    
#     # Perform pairwise t-tests based on Levene’s test result
#     for i in range(len(diagnoses)):
#         for j in range(i + 1, len(diagnoses)):
#             d1, d2 = diagnoses[i], diagnoses[j]
#             group1 = df[df['diagnosis'] == d1][metric].dropna()
#             group2 = df[df['diagnosis'] == d2][metric].dropna()

#             if len(group1) > 1 and len(group2) > 1:
#                 if variance_results[(d1, d2)] > 0.05:
#                     # If p-value > 0.05 from Levene's test, assume equal variance (use standard t-test)
#                     t_stat, p_val = ttest_ind(group1, group2, equal_var=True)
#                     test_type = 'Standard t-test'
#                 else:
#                     # If p-value <= 0.05 from Levene's test, assume unequal variance (use Welch's t-test)
#                     t_stat, p_val, _ = welch_ttest(group1, group2, usevar='unequal')
#                     test_type = 'Welch’s t-test'
                
#                 print(f'Between {d1} and {d2} ({test_type}): t-stat = {t_stat:.4f}, p-val = {p_val:.4f}')
    
#     return anova_table, diagnoses

# # def run_anova_and_ttests(df, metric):
# #     formula = f'{metric} ~ C(diagnosis)'
# #     model = ols(formula, data=df).fit()
# #     anova_table = sm.stats.anova_lm(model, typ=2)
# #     print(f'\nANOVA for {metric}:\n', anova_table)

# #     print(f'\nPairwise t-tests for {metric}:')
# #     diagnoses = sorted(df['diagnosis'].unique())
# #     for i in range(len(diagnoses)):
# #         for j in range(i+1, len(diagnoses)):
# #             d1, d2 = diagnoses[i], diagnoses[j]
# #             group1 = df[df['diagnosis'] == d1][metric].dropna()
# #             group2 = df[df['diagnosis'] == d2][metric].dropna()
# #             if len(group1) > 1 and len(group2) > 1:
# #                 t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
# #                 print(f'Between {d1} and {d2}: t-stat = {t_stat:.4f}, p-val = {p_val:.4f}')

# #     return anova_table, diagnoses

# def plot_bar_graph(df, metric, anova_result, diagnoses, dest_folder):
#     plt.figure(figsize=(12, 8))
#     sns.set(style="whitegrid")

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(diagnoses), CUSTOM_COLORS)}

#     # Create barplot without hue
#     ax = sns.barplot(x='diagnosis', y=metric, data=df, errorbar=('ci',95), palette=palette, order=(diagnoses))

#     # Adjust y-axis to start slightly above 0
#     y_min = df[metric].min() * 0.95  # 5% below the minimum value
#     y_max = df[metric].max() * 1.2   # 40% above the maximum value to accommodate significance lines
#     ax.set_ylim(y_min, y_max)

#     # Add y-axis break indication
#     # Using a diagonal line to indicate the axis does not start at zero
#     d = .015  # size of diagonal lines
#     kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#     ax.plot((-d, +d), (-d, +d), **kwargs)        # bottom-left diagonal
#     # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-right diagonal

#     ax.set_xlabel('Diagnosis', fontsize=14)
#     ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
#     ax.set_title(f'{metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

#     # Add significance annotations
#     significance_lines = []
#     for i in range(len(diagnoses)):
#         for j in range(i+1, len(diagnoses)):
#             d1, d2 = sorted(diagnoses)[i], sorted(diagnoses)[j]
#             group1 = df[df['diagnosis'] == d1][metric].dropna()
#             group2 = df[df['diagnosis'] == d2][metric].dropna()
#             if len(group1) > 1 and len(group2) > 1:
#                 _, p_val = ttest_ind(group1, group2, equal_var=False)
#                 if p_val < 0.05:
#                     significance_lines.append((i, j, p_val))

#     # Sort significance lines by p-value for better spacing
#     significance_lines = sorted(significance_lines, key=lambda x: x[2])

#     # Initialize base y position for annotations
#     base_y = y_max * 0.07  # Starting 7% above the top bar
#     line_height = y_max * 0.02  # Height of each significance line
#     increment = y_max * 0.05  # Increment for each additional line

#     for idx, (i, j, p_val) in enumerate(significance_lines):
#         y = y_max - base_y - (idx * increment)
#         ax.plot([i, i, j, j], [y, y + line_height, y + line_height, y], lw=1.5, color='k')
#         ax.text((i + j) / 2, y + line_height, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=12)

#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{metric}_bar_graph.png'), dpi=300)
#     plt.close()

# def plot_violin_plot(df, metric, dest_folder):
#     plt.figure(figsize=(12, 8))
#     sns.set(style="whitegrid")

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

#     ax = sns.violinplot(x='diagnosis', y=metric, data=df, palette=palette, order=sorted(df['diagnosis'].unique()))

#     ax.set_xlabel('Diagnosis', fontsize=14)
#     ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
#     ax.set_title(f'Distribution of {metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{metric}_violin_plot.png'), dpi=300)
#     plt.close()

# def plot_scatter_with_regression(df, x_metric, y_metric, dest_folder):
#     plt.figure(figsize=(12, 8))
#     sns.set(style="whitegrid")

#     # Only include rows where both metrics are present
#     subset = df.dropna(subset=[x_metric, y_metric])

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(subset['diagnosis'].unique()), CUSTOM_COLORS)}

#     ax = sns.scatterplot(x=x_metric, y=y_metric, hue='diagnosis', data=subset, palette=palette)
#     y_min = 0  # 5% below the minimum value
#     y_max = df[y_metric].max() * 1.2   # 40% above the maximum value to accommodate significance lines
#     ax.set_ylim(y_min, y_max)
#     # Fit and plot regression lines for each diagnosis
#     diagnoses = sorted(subset['diagnosis'].unique())
#     for diagnosis in diagnoses:
#         subset_diag = subset[subset['diagnosis'] == diagnosis]
#         if len(subset_diag) > 1:
#             sns.regplot(x=x_metric, y=y_metric, data=subset_diag, scatter=False, label=diagnosis, color=palette[diagnosis])

#     ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=14)
#     ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=14)
#     ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}', fontsize=16)
#     ax.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')

#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{y_metric}_vs_{x_metric}_scatter.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_correlation_matrix(df, metrics, dest_folder):
#     plt.figure(figsize=(10, 8))
#     sns.set(style="white")

#     # Select relevant metrics and drop rows with missing values
#     corr_df = df[metrics].dropna()

#     corr_matrix = corr_df.corr()

#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)

#     plt.title('Correlation Matrix', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, 'correlation_matrix.png'), dpi=300)
#     plt.close()

# def plot_box_plot(df, metric, dest_folder):
#     plt.figure(figsize=(12, 8))
#     sns.set(style="whitegrid")

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

#     ax = sns.boxplot(x='diagnosis', y=metric, data=df, palette=palette, order=sorted(df['diagnosis'].unique()))

#     ax.set_xlabel('Diagnosis', fontsize=14)
#     ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
#     ax.set_title(f'Box Plot of {metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{metric}_box_plot.png'), dpi=300)
#     plt.close()

# def plot_histogram(df, metric, dest_folder):
#     plt.figure(figsize=(12, 8))
#     sns.set(style="whitegrid")

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

#     sns.histplot(data=df, x=metric, hue='diagnosis', multiple='stack', palette=palette, kde=True)

#     plt.xlabel(metric.replace('_', ' ').title(), fontsize=14)
#     plt.ylabel('Count', fontsize=14)
#     plt.title(f'Histogram of {metric.replace("_", " ").title()}', fontsize=16)

#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{metric}_histogram.png'), dpi=300)
#     plt.close()

# def plot_correlation_scatter(df, metric_x, metric_y, dest_folder):
#     plt.figure(figsize=(10, 8))
#     sns.set(style="whitegrid")

#     # Map diagnoses to colors
#     palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

#     sns.scatterplot(x=metric_x, y=metric_y, hue='diagnosis', data=df, palette=palette)

#     plt.xlabel(metric_x.replace('_', ' ').title(), fontsize=14)
#     plt.ylabel(metric_y.replace('_', ' ').title(), fontsize=14)
#     plt.title(f'{metric_y.replace("_", " ").title()} vs {metric_x.replace("_", " ").title()}', fontsize=16)

#     plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(dest_folder, f'{metric_y}_vs_{metric_x}_correlation.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main(mnd_csv_file, control_csv_file,control_csv_file_1, src_root_mnd, src_root_control, dest_folder):
#     os.makedirs(dest_folder, exist_ok=True)

#     print("Processing MND files...")
#     mnd_df = process_mnd_files(mnd_csv_file, src_root_mnd)
#     print(f"MND data: {mnd_df.shape[0]} records")

#     print("Processing Control files...")
#     control_df = process_control_files(control_csv_file, src_root_control)
#     print(f"Control data: {control_df.shape[0]} records")
    
#     print("Combining datasets...")
#     control_df_1 = process_control_files(control_csv_file_1,src_root_control)
#     df = pd.concat([mnd_df, control_df], ignore_index=True)
#     df = pd.concat([df, control_df_1], ignore_index=True)
#     # df = pd.concat([mnd_df, control_df], ignore_index=True)

#     # Convert relevant columns to numeric
#     numeric_cols = ['paracentral', 'precentral', 'cortical_thickness', 'age_at_scan', 'months_since_onset']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     # Define metrics for analysis
#     metrics = ['cortical_thickness', 'paracentral', 'precentral']

#     # Run ANOVA and T-tests, then plot bar graphs and other plots
#     for metric in metrics:
#         print(f"\nAnalyzing {metric}...")
#         anova_result, diagnoses = run_anova_and_ttests(df, metric)
#         plot_bar_graph(df, metric, anova_result, diagnoses, dest_folder)
#         plot_violin_plot(df, metric, dest_folder)
#         plot_box_plot(df, metric, dest_folder)
#         plot_histogram(df, metric, dest_folder)

#     # Plot scatter plots
#     if 'months_since_onset' in df.columns:
#         plot_scatter_with_regression(df, 'months_since_onset', 'cortical_thickness', dest_folder)
#         plot_scatter_with_regression(df, 'months_since_onset', 'paracentral', dest_folder)
#         plot_scatter_with_regression(df, 'months_since_onset', 'precentral', dest_folder)

#     # Plot correlation matrix
#     plot_correlation_matrix(df, metrics + ['age_at_scan', 'months_since_onset'], dest_folder)

#     # Additional correlation scatter plots
#     plot_correlation_scatter(df, 'paracentral', 'precentral', dest_folder)
#     plot_correlation_scatter(df, 'paracentral', 'cortical_thickness', dest_folder)
#     plot_correlation_scatter(df, 'precentral', 'cortical_thickness', dest_folder)

#     print("All analyses and plots are completed.")

# # Define source and destination paths
# mnd_csv_file = '/home/groups/dlmrimnd/akshit/New_MND_patient_data_summary.csv'
# control_csv_file = '/home/groups/dlmrimnd/akshit/NEW_ADNI_Normal_control_patient_data_summary.csv'
# control_csv_file_1 = '/home/groups/dlmrimnd/akshit/Normal_control_patient_data_summary.csv'
# src_root_mnd = '/home/groups/dlmrimnd/akshit/MND_patients/'
# src_root_control = '/home/groups/dlmrimnd/akshit/Normal_control/'
# dest_folder = '/home/groups/dlmrimnd/akshit/geometricDL_MND/images/adv_graphs/'
# # dest_folder = '/home/groups/dlmrimnd/akshit/Graph/'


# # Run the main function
# if __name__ == "__main__":
#     main(mnd_csv_file, control_csv_file,control_csv_file_1,src_root_mnd, src_root_control, dest_folder)






# without other 
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import f_oneway, ttest_ind, levene
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ttest_ind as welch_ttest

# Define diagnosis mapping
DIAGNOSIS_MAPPING = {
    0: 'Normal Control',
    1: 'ALS',
    2: 'PLS',
    4: 'PMA'
}

# Define custom color palette
CUSTOM_COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

def extract_data(file_path):
    paracentral_vol = None
    precentral_vol = None
    etiv = None
    cortical_thickness = None

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

def process_mnd_files(csv_file, src_root_mnd):
    data = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            if folder == str(108):
                continue  # Skip folder 108 as per user request
            diagnosis_num = row.get('formal.diagnosis.numeric', row.get('label', '0'))
            try:
                diagnosis_num = int(diagnosis_num)
            except ValueError:
                diagnosis_num = 0  # Default to control if invalid

            diagnosis = DIAGNOSIS_MAPPING.get(diagnosis_num, 'Other')

            # Only include desired diagnoses
            if diagnosis not in ['Normal Control', 'ALS', 'PLS', 'PMA']:
                continue  # Skip this record

            folder_path = os.path.join(src_root_mnd, folder)

            stats_dir = os.path.join(folder_path, 'stats')
            lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
            rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

            if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
                lh_paracentral_vol, lh_precentral_vol, lh_etiv, lh_cortical_thickness = extract_data(lh_stats_file)
                rh_paracentral_vol, rh_precentral_vol, rh_etiv, rh_cortical_thickness = extract_data(rh_stats_file)

                if lh_etiv and rh_etiv:
                    etiv = lh_etiv  # Assuming ETIV is the same for both hemispheres
                    paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
                    precentral_vol = lh_precentral_vol + rh_precentral_vol
                    cortical_thickness = (lh_cortical_thickness + rh_cortical_thickness) / 2  # Average

                    months_since_onset = row.get('months.since.onset', np.nan)
                    try:
                        months_since_onset = float(months_since_onset)
                    except ValueError:
                        months_since_onset = np.nan

                    data.append({
                        'folder': folder,
                        'diagnosis_num': diagnosis_num,
                        'diagnosis': diagnosis,
                        'paracentral': paracentral_vol / etiv,
                        'precentral': precentral_vol / etiv,
                        'cortical_thickness': cortical_thickness,
                        'age_at_scan': row.get('age.at.scan', np.nan),
                        'sex': row.get('sex.numerical', np.nan),
                        'months_since_onset': months_since_onset
                    })

    return pd.DataFrame(data)

def process_control_files(csv_file, src_root_control):
    control_data = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row['folder']
            diagnosis_num = int(row.get('label', 0))
            diagnosis = DIAGNOSIS_MAPPING.get(diagnosis_num, 'Normal Control')  # Assuming label=0 is control

            # Only include desired diagnoses
            if diagnosis not in ['Normal Control', 'ALS', 'PLS', 'PMA']:
                continue  # Skip this record

            folder_path = os.path.join(src_root_control, folder)

            stats_dir = os.path.join(folder_path, 'stats')
            lh_stats_file = os.path.join(stats_dir, 'lh.aparc.DKTatlas.mapped.stats')
            rh_stats_file = os.path.join(stats_dir, 'rh.aparc.DKTatlas.mapped.stats')

            if os.path.exists(lh_stats_file) and os.path.exists(rh_stats_file):
                lh_paracentral_vol, lh_precentral_vol, lh_etiv, lh_cortical_thickness = extract_data(lh_stats_file)
                rh_paracentral_vol, rh_precentral_vol, rh_etiv, rh_cortical_thickness = extract_data(rh_stats_file)

                if lh_etiv and rh_etiv:
                    etiv = lh_etiv  # Assuming ETIV is the same for both hemispheres
                    paracentral_vol = lh_paracentral_vol + rh_paracentral_vol
                    precentral_vol = lh_precentral_vol + rh_precentral_vol
                    cortical_thickness = (lh_cortical_thickness + rh_cortical_thickness) / 2  # Average

                    age_at_scan = row.get('age.at.scan', np.nan)
                    try:
                        age_at_scan = float(age_at_scan)
                    except ValueError:
                        age_at_scan = np.nan

                    sex = row.get('sex.numerical', np.nan)
                    try:
                        sex = int(sex)
                    except ValueError:
                        sex = np.nan

                    control_data.append({
                        'folder': folder,
                        'diagnosis_num': diagnosis_num,
                        'diagnosis': diagnosis,
                        'paracentral': paracentral_vol / etiv,
                        'precentral': precentral_vol / etiv,
                        'cortical_thickness': cortical_thickness,
                        'age_at_scan': age_at_scan,
                        'sex': sex,
                        'months_since_onset': np.nan  # Not applicable for controls
                    })

    return pd.DataFrame(control_data)

def run_anova_and_ttests(df, metric):
    formula = f'{metric} ~ C(diagnosis)'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f'\nANOVA for {metric}:\n', anova_table)

    print(f'\nTesting for equal variances and performing pairwise t-tests for {metric}:')
    diagnoses = sorted(df['diagnosis'].unique())
    variance_results = {}
    
    # Perform Levene’s test to check for equal variances
    for i in range(len(diagnoses)):
        for j in range(i + 1, len(diagnoses)):
            d1, d2 = diagnoses[i], diagnoses[j]
            group1 = df[df['diagnosis'] == d1][metric].dropna()
            group2 = df[df['diagnosis'] == d2][metric].dropna()
            
            if len(group1) > 1 and len(group2) > 1:
                # Levene's test
                stat, p = levene(group1, group2)
                variance_results[(d1, d2)] = p
                print(f'Levene’s test between {d1} and {d2}: p-value = {p:.4f}')
    
    # Perform pairwise t-tests based on Levene’s test result
    for i in range(len(diagnoses)):
        for j in range(i + 1, len(diagnoses)):
            d1, d2 = diagnoses[i], diagnoses[j]
            group1 = df[df['diagnosis'] == d1][metric].dropna()
            group2 = df[df['diagnosis'] == d2][metric].dropna()

            if len(group1) > 1 and len(group2) > 1:
                if variance_results[(d1, d2)] > 0.05:
                    # If p-value > 0.05 from Levene's test, assume equal variance (use standard t-test)
                    t_stat, p_val = ttest_ind(group1, group2, equal_var=True)
                    test_type = 'Standard t-test'
                else:
                    # If p-value <= 0.05 from Levene's test, assume unequal variance (use Welch's t-test)
                    t_stat, p_val, _ = welch_ttest(group1, group2, usevar='unequal')
                    test_type = 'Welch’s t-test'
                
                print(f'Between {d1} and {d2} ({test_type}): t-stat = {t_stat:.4f}, p-val = {p_val:.4f}')
    
    return anova_table, diagnoses

def plot_bar_graph(df, metric, anova_result, diagnoses, dest_folder):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(diagnoses), CUSTOM_COLORS)}

    # Create barplot without hue
    ax = sns.barplot(x='diagnosis', y=metric, data=df, errorbar=('ci',95), palette=palette, order=(diagnoses))

    # Adjust y-axis to start slightly above 0
    y_min = df[metric].min() * 0.95  # 5% below the minimum value
    y_max = df[metric].max() * 1.2   # 20% above the maximum value to accommodate significance lines
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel('Diagnosis', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'{metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

    # Add significance annotations
    significance_lines = []
    for i in range(len(diagnoses)):
        for j in range(i+1, len(diagnoses)):
            d1, d2 = diagnoses[i], diagnoses[j]
            group1 = df[df['diagnosis'] == d1][metric].dropna()
            group2 = df[df['diagnosis'] == d2][metric].dropna()
            if len(group1) > 1 and len(group2) > 1:
                _, p_val = ttest_ind(group1, group2, equal_var=False)
                if p_val < 0.05:
                    significance_lines.append((i, j, p_val))

    # Sort significance lines by p-value for better spacing
    significance_lines = sorted(significance_lines, key=lambda x: x[2])

    # Initialize base y position for annotations
    base_y = y_max * 0.07  # Starting 7% above the top bar
    line_height = y_max * 0.02  # Height of each significance line
    increment = y_max * 0.05  # Increment for each additional line

    for idx, (i, j, p_val) in enumerate(significance_lines):
        y = y_max - base_y - (idx * increment)
        ax.plot([i, i, j, j], [y, y + line_height, y + line_height, y], lw=1.5, color='k')
        ax.text((i + j) / 2, y + line_height, f'p={p_val:.3f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{metric}_bar_graph.png'), dpi=300)
    plt.close()

def plot_violin_plot(df, metric, dest_folder):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

    ax = sns.violinplot(x='diagnosis', y=metric, data=df, palette=palette, order=sorted(df['diagnosis'].unique()))

    ax.set_xlabel('Diagnosis', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'Distribution of {metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{metric}_violin_plot.png'), dpi=300)
    plt.close()

def plot_scatter_with_regression(df, x_metric, y_metric, dest_folder):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Only include rows where both metrics are present
    subset = df.dropna(subset=[x_metric, y_metric])

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(subset['diagnosis'].unique()), CUSTOM_COLORS)}

    ax = sns.scatterplot(x=x_metric, y=y_metric, hue='diagnosis', data=subset, palette=palette)
    y_min = 0
    y_max = df[y_metric].max() * 1.2
    ax.set_ylim(y_min, y_max)

    # Fit and plot regression lines for each diagnosis
    diagnoses = sorted(subset['diagnosis'].unique())
    for diagnosis in diagnoses:
        subset_diag = subset[subset['diagnosis'] == diagnosis]
        if len(subset_diag) > 1:
            sns.regplot(x=x_metric, y=y_metric, data=subset_diag, scatter=False, label=diagnosis, color=palette[diagnosis])

    ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}', fontsize=16)
    ax.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{y_metric}_vs_{x_metric}_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df, metrics, dest_folder):
    plt.figure(figsize=(10, 8))
    sns.set(style="white")

    # Select relevant metrics and drop rows with missing values
    corr_df = df[metrics].dropna()

    corr_matrix = corr_df.corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)

    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, 'correlation_matrix.png'), dpi=300)
    plt.close()

def plot_box_plot(df, metric, dest_folder):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

    ax = sns.boxplot(x='diagnosis', y=metric, data=df, palette=palette, order=sorted(df['diagnosis'].unique()))

    ax.set_xlabel('Diagnosis', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'Box Plot of {metric.replace("_", " ").title()} by Diagnosis', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{metric}_box_plot.png'), dpi=300)
    plt.close()

def plot_histogram(df, metric, dest_folder):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

    sns.histplot(data=df, x=metric, hue='diagnosis', multiple='stack', palette=palette, kde=True)

    plt.xlabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.title(f'Histogram of {metric.replace("_", " ").title()}', fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{metric}_histogram.png'), dpi=300)
    plt.close()

def plot_correlation_scatter(df, metric_x, metric_y, dest_folder):
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # Map diagnoses to colors
    palette = {diagnosis: color for diagnosis, color in zip(sorted(df['diagnosis'].unique()), CUSTOM_COLORS)}

    sns.scatterplot(x=metric_x, y=metric_y, hue='diagnosis', data=df, palette=palette)

    plt.xlabel(metric_x.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(metric_y.replace('_', ' ').title(), fontsize=14)
    plt.title(f'{metric_y.replace("_", " ").title()} vs {metric_x.replace("_", " ").title()}', fontsize=16)

    plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(dest_folder, f'{metric_y}_vs_{metric_x}_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main(mnd_csv_file, control_csv_file, control_csv_file_1, src_root_mnd, src_root_control, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    print("Processing MND files...")
    mnd_df = process_mnd_files(mnd_csv_file, src_root_mnd)
    print(f"MND data: {mnd_df.shape[0]} records")

    print("Processing Control files...")
    control_df = process_control_files(control_csv_file, src_root_control)
    print(f"Control data: {control_df.shape[0]} records")
    
    print("Processing additional Control files...")
    control_df_1 = process_control_files(control_csv_file_1, src_root_control)
    print(f"Additional Control data: {control_df_1.shape[0]} records")

    print("Combining datasets...")
    df = pd.concat([mnd_df, control_df], ignore_index=True)
    df = pd.concat([df, control_df_1], ignore_index=True)

    # Filter df to include only desired diagnoses
    desired_diagnoses = ['Normal Control', 'ALS', 'PLS', 'PMA']
    df = df[df['diagnosis'].isin(desired_diagnoses)]

    # Convert relevant columns to numeric
    numeric_cols = ['paracentral', 'precentral', 'cortical_thickness', 'age_at_scan', 'months_since_onset']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Define metrics for analysis
    metrics = ['cortical_thickness', 'paracentral', 'precentral']

    # Run ANOVA and T-tests, then plot bar graphs and other plots
    for metric in metrics:
        print(f"\nAnalyzing {metric}...")
        anova_result, diagnoses = run_anova_and_ttests(df, metric)
        plot_bar_graph(df, metric, anova_result, diagnoses, dest_folder)
        plot_violin_plot(df, metric, dest_folder)
        plot_box_plot(df, metric, dest_folder)
        plot_histogram(df, metric, dest_folder)

    # Plot scatter plots
    if 'months_since_onset' in df.columns:
        plot_scatter_with_regression(df, 'months_since_onset', 'cortical_thickness', dest_folder)
        plot_scatter_with_regression(df, 'months_since_onset', 'paracentral', dest_folder)
        plot_scatter_with_regression(df, 'months_since_onset', 'precentral', dest_folder)

    # Plot correlation matrix
    plot_correlation_matrix(df, metrics + ['age_at_scan', 'months_since_onset'], dest_folder)

    # Additional correlation scatter plots
    plot_correlation_scatter(df, 'paracentral', 'precentral', dest_folder)
    plot_correlation_scatter(df, 'paracentral', 'cortical_thickness', dest_folder)
    plot_correlation_scatter(df, 'precentral', 'cortical_thickness', dest_folder)

    print("All analyses and plots are completed.")

# Define source and destination paths
mnd_csv_file = '/home/groups/dlmrimnd/akshit/New_MND_patient_data_summary.csv'
control_csv_file = '/home/groups/dlmrimnd/akshit/NEW_ADNI_Normal_control_patient_data_summary.csv'
control_csv_file_1 = '/home/groups/dlmrimnd/akshit/Normal_control_patient_data_summary.csv'
src_root_mnd = '/home/groups/dlmrimnd/akshit/MND_patients/'
src_root_control = '/home/groups/dlmrimnd/akshit/Normal_control/'
dest_folder = '/home/groups/dlmrimnd/akshit/geometricDL_MND/images/adv_graphs/'
# dest_folder = '/home/groups/dlmrimnd/akshit/Graph/'

# Run the main function
if __name__ == "__main__":
    main(mnd_csv_file, control_csv_file, control_csv_file_1, src_root_mnd, src_root_control, dest_folder)
