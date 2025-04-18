#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:18:24 2024

@author: pierrebeaucoral
"""

import pandas as pd
import os

# Step 1: Set up the working directory
wd = "/UNDERCANOPY-main/Climate finance estimation/"
os.chdir(wd)

# Step 2: Load the first CSV file with low_memory=False to prevent DtypeWarnings
try:
    df1 = pd.read_csv(os.path.join(wd,'Data/projects_clusters.csv'))
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Step 2.5: Create a filtered DataFrame for Climate Adaptation projects where Topic is equal to 5
climate_adaptation = df1[df1['Topic'] == 5].copy().reset_index(drop=True)

# Optionally, save the climate_adaptation DataFrame to a CSV file for future reference
climate_adaptation_output_path = os.path.join(wd,'Data/climate_adaptation_projects.csv')
climate_adaptation.to_csv(climate_adaptation_output_path, index=False)

print(f"Climate Adaptation DataFrame saved to {climate_adaptation_output_path}")

# Step 3: Filter the DataFrame based on specific values in 'Topic'
Climate_Topics = [2, 5, 11, 26, 29, 43, 89, 123, 129, 162, 165, 213, 235, 256, 263, 269, 276, 281, 290, 312, 335, 338, 361, 366, 385]
filtered_df1 = df1[df1['Topic'].isin(Climate_Topics)].copy()

# Save the projects that are not in Climate_Topics
non_climate_df1 = df1[~df1['Topic'].isin(Climate_Topics)].drop_duplicates(subset='raw_text').copy()
non_climate_output_path = './Data/non_climate_projects.csv'
non_climate_df1.to_csv(non_climate_output_path, index=False)

# Step 4: Rename 'CustomName' based on 'Topic' values and add 'relevance' column
rename_dict = {
    11: "Renewable energy", 312: "Renewable energy",
    123: "Wildlife conservation", 129: "Wildlife conservation",
    162: "Wildlife conservation", 235: "Wildlife conservation",
    256: "Wildlife conservation", 269: "Wildlife conservation",
    338: "Wildlife conservation", 192: "Resilience",
    276: "Resilience", 366: "Geothermal Explr/Plants",
    361: "Geothermal Explr/Plants" 
}

if 'CustomName' in filtered_df1.columns:
    filtered_df1['CustomName'] = filtered_df1['Topic'].map(rename_dict).fillna(filtered_df1['CustomName'])
else:
    print("Warning: 'CustomName' column not found in the DataFrame.")

filtered_df1['relevance'] = 1

climate_projects_output_path = os.path.join(wd,'Data/climate_projects.csv')
filtered_df1.to_csv(climate_projects_output_path, index=False)

# Step 5: Remove duplicates based on the 'raw_text' column
filtered_df1 = filtered_df1.drop_duplicates(subset='raw_text')

# Step 6: Calculate the total sample size (e.g., 1% of the total dataset)
total_sample_size = int(0.05 * len(filtered_df1))

# Step 7: Determine the proportionate sample size for each 'Topic' group
topic_group_sizes = filtered_df1.groupby('Topic').size()
topic_proportions = topic_group_sizes / len(filtered_df1)
topic_sample_sizes = (topic_proportions * total_sample_size).round().astype(int)

# Step 8: Sample from each 'Topic' group using the calculated sample sizes
sampled_df1 = filtered_df1.groupby('Topic', group_keys=False).apply(
    lambda x: x.sample(n=topic_sample_sizes.loc[x.name], random_state=42)
)

# Step 9: Keep only the columns 'raw_text', 'relevance', and 'CustomName'
sampled_df1 = sampled_df1[['raw_text', 'relevance', 'CustomName']]

# Step 10: Rename 'raw_text' to 'text' and 'CustomName' to 'label'
sampled_df1 = sampled_df1.rename(columns={'raw_text': 'text', 'CustomName': 'label'})

# Step 11: Load the second CSV file
try:
    df2 = pd.read_csv(os.path.join(wd, 'Data/train_set.csv'), delimiter=";")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    raise

# Step 12: Filter the second DataFrame to keep only rows where 'relevance' is 0
filtered_df2 = df2[df2['relevance'] == 0].copy()

# Step 13: Merge the two DataFrames
merged_df = pd.concat([sampled_df1, filtered_df2])

# Step 14: Sort the merged DataFrame by the 'text' variable and reset the index
merged_df = merged_df.sort_values(by='text').reset_index(drop=True)

# Step 15: Replace projects in merged_df with projects from climate_adaptation based on their index

# List of indices to remove from merged_df
indices_to_remove = [125, 132, 140, 163, 206, 239, 241, 277, 332, 337, 410, 577, 585, 659, 685, 734, 770, 782,
                     868, 875, 977, 978, 984, 989, 990, 991, 998, 1027, 1055, 1064, 1098, 1164, 1180, 1183, 1188,
                     1190, 1194, 1199, 1200, 1201, 1202, 1203, 1208, 1210, 1216, 1224, 1226, 1233, 1249, 1297, 1371,
                     1462, 1463, 1466, 1490, 1498, 1500, 1505, 1511, 1522, 1545, 1546, 1556, 1557, 1636, 1637, 1638,
                     1647, 1717, 1773, 1816, 1894, 1902, 2171, 2172, 2295, 2299, 2342, 2347, 2348, 2349, 2350, 2368,
                     2373, 2375, 2382, 2388, 2392, 2396, 2428, 2429, 2443, 2482, 2493, 2525, 2629, 2539, 2548, 2561,
                     2563, 2564, 2653, 2687, 2706, 2774, 2775, 2776, 2777, 2778, 2781, 2791, 2867, 2940, 3050, 3623,  
                     3784, 3791, 3792, 3793, 3806, 3820, 4070, 4083, 4129, 4132, 4134, 4153, 4294, 4296, 4328, 4345,
                     4356, 4363, 4364, 4725, 4762, 5067, 5146, 5154, 5233, 5234, 5290, 5415, 5453, 5455, 5636, 5671,
                     5674, 5762, 5772, 5780, 5927, 5976, 6205, 6286, 6296, 6308, 6393, 6439, 6440, 6470, 6477, 6482,
                     6590, 6630, 6633, 6634, 6653 
                     ]

# List of indices to add from climate_adaptation
indices_to_add = [651748, 1285280, 1287267, 1288499, 346781, 346841, 354897, 357084, 366546, 402437, 405866, 431985,
                  433174, 435298, 450005, 451438, 461620, 462799, 465258, 465288, 473057, 476155, 476994, 478814, 
                  479305, 481934, 484113, 484276, 484403, 484959, 488044, 488155, 493188, 494872, 497316, 497427,
                  502986, 503021, 504412, 504736, 519972, 519989, 520784, 520813, 521148, 521760, 522135, 522165,
                  523368, 523770, 523387, 538987, 542408, 542430, 542456, 542483, 542521, 542611, 553096, 557824,
                  558460, 558724, 558724, 559820, 563273, 568582, 569294, 570989, 571164, 571353, 572335, 572399, 
                  572482, 578446, 582516, 582848, 583167, 583174, 583276, 583414, 585113, 591035, 591222, 591755,
                  591807, 593384, 594848, 597285, 595860, 598968, 600487, 601250, 602696, 602838, 603732, 603752,
                  605102, 605217, 607906, 608081, 608128, 608396, 608799, 609131, 609398, 609807, 610528, 611144,
                  613099, 613292, 613770, 615225, 615743, 618314, 621645, 622880, 623146, 623329, 625104, 628391,
                  631711, 633308, 633946, 634610, 634615, 634619, 635619, 642179, 648267, 651770, 651777, 651789, 
                  651790, 651794, 653144, 656383, 657976, 660279, 661142, 661985, 666410, 666533, 666955, 666988, 
                  667013, 667672, 669885, 671915, 652561, 672642, 673245, 674040, 674246, 677620, 679361, 682680,
                  686116, 690499, 690812
                   ]

# Define a dictionary where keys are the labels and values are lists of indices
label_updates = {
    "Hydro Power Plants Rehab":[3453, 4908],
    'Forest Sustainability: Tropical, Sustainable Management, Deforestation, REDD+': [402, 4454, 5637],
    'Air Pollution Mitigation': [1217, 1258, 1259, 1262, 2775, 2776, 2777, 2778, 2809, 2934, 3044, 3145, 3370,
                                 4328, 4426, 4844, 5620, 5852],
    'Renewable energy': [1113, 1181, 1251, 1253, 1585, 2803, 3963, 4134, 4583, 5444, 5445, 5685, 5727],
    'Biodiv Conserv Prog': [803, 2773, 3378, 4504, 4505, 5082, 5414, 5836],
    'Solar PV Energy': [2789],
    'Geothermal Explr/Plants': [],
    'Wildlife conservation':[788],
    'Wind power farms': [], 
    'Environmental Policy Admin': [1228, 1229, 2206, 2236, 3327, 3328, 3458, 3459, 3696, 3698, 3728, 5397, 5684,
                                   5721],
    'Green Growth Strategies': [626, 2457, 2798, 2799, 2891, 3166, 3546, 3837, 6453, 6623],
    'Marine-Coastal Protected Areas Mgmt. (CMB)':[3053],
    'Enviro Ed Trainings': [1254, 1939, 2774, 4376, 5057],
    'Combat Desertif Convention': [3456, 3457],
    'Climate Adaptation': [352, 1934, 3045, 4541, 4843, 5683],
    'National Capacities - Enviro Dev Plan Mainstreaming': [3082],
    'Resilience': [552, 909, 1191, 5844, 6178, 6594]
}

# Iterate over the dictionary and update the 'label' and 'relevance' columns accordingly
for label, indices in label_updates.items():
    # Update the 'label' column
    merged_df.loc[indices, 'label'] = label
    # Update the 'relevance' column to 1
    merged_df.loc[indices, 'relevance'] = 1


# Select only the 'raw_text' and 'CustomName' columns from projects_to_add
projects_to_add = df1.iloc[indices_to_add][['raw_text', 'CustomName']].copy()

# Rename 'raw_text' to 'text' and 'CustomName' to 'label'
projects_to_add = projects_to_add.rename(columns={'raw_text': 'text', 'CustomName': 'label'})

# Set 'relevance' to 1 for the added projects
projects_to_add['relevance'] = 1

# Now remove the projects from merged_df based on the indices to remove
merged_df.drop(index=indices_to_remove, inplace=True)

# Add the projects from climate_adaptation to merged_df
merged_df = pd.concat([merged_df, projects_to_add], ignore_index=True)

# Reset the index of merged_df after removing and adding projects
merged_df = merged_df.reset_index(drop=True)

# Step 16: Remove duplicates based on 'text'
before_deduplication = len(merged_df)  # Record the number of rows before deduplication
merged_df = merged_df.drop_duplicates(subset='text')
after_deduplication = len(merged_df)  # Record the number of rows after deduplication

# Step 17: Print the number of duplicates removed
duplicates_removed = before_deduplication - after_deduplication
print(f"Number of duplicates removed: {duplicates_removed}")

# Step 18: Check for duplicates in newly added projects based on 'text'
# Compare the 'text' column of the added projects
duplicates_in_added_projects = projects_to_add.duplicated(subset='text', keep=False)

if duplicates_in_added_projects.any():
    print("Duplicates found in newly added projects:")
    print(projects_to_add[duplicates_in_added_projects])
else:
    print("No duplicates found in the newly added projects.")
    
# Keep only 'text', 'label', and 'relevance' columns
merged_df = merged_df[['text', 'label', 'relevance']]

# Step 19: Calculate the number of projects with relevance == 1
num_relevance_1 = len(merged_df[merged_df['relevance'] == 1])

# Step 20: Calculate the number of projects needed from non_climate_df1 to balance the dataset
# We'll need to add this many rows with relevance == 0
num_needed_relevance_0 = num_relevance_1 - len(merged_df[merged_df['relevance'] == 0])

# Step 21: Remove duplicates from non_climate_df1 that already exist in merged_df (based on 'text')
non_climate_df1_unique = non_climate_df1[~non_climate_df1['raw_text'].isin(merged_df['text'])]

# Step 22: Calculate the proportion of each non-climate Topic in non_climate_df1_unique
non_climate_topic_sizes = non_climate_df1_unique.groupby('Topic').size()
non_climate_topic_proportions = non_climate_topic_sizes / len(non_climate_df1_unique)

# Step 23: Calculate the number of samples to take from each non-climate Topic based on the proportions
non_climate_sample_sizes = (non_climate_topic_proportions * num_needed_relevance_0).round().astype(int)

# Step 24: Sample projects from non_climate_df1 according to the calculated sample sizes
sampled_non_climate_df1 = non_climate_df1_unique.groupby('Topic', group_keys=False).apply(
    lambda x: x.sample(n=non_climate_sample_sizes.loc[x.name], random_state=42)
)

# Step 25: Keep only the relevant columns from sampled_non_climate_df1
sampled_non_climate_df1 = sampled_non_climate_df1[['raw_text', 'CustomName']].rename(
    columns={'raw_text': 'text', 'CustomName': 'label'}
)

# Step 26: Set relevance to 0 for the sampled non-climate projects
sampled_non_climate_df1['relevance'] = 0

# Step 27: Add the sampled non-climate projects to merged_df
merged_df = pd.concat([merged_df, sampled_non_climate_df1], ignore_index=True)

# Step 28: Re-check for duplicates in merged_df after adding non-climate projects
before_deduplication = len(merged_df)
merged_df = merged_df.drop_duplicates(subset='text')
after_deduplication = len(merged_df)

# Step 29: Print the number of duplicates removed
duplicates_removed = before_deduplication - after_deduplication
print(f"Number of duplicates removed after adding non-climate projects: {duplicates_removed}")

# Step 30: Ensure the final DataFrame is balanced and save it
num_relevance_1_final = len(merged_df[merged_df['relevance'] == 1])
num_relevance_0_final = len(merged_df[merged_df['relevance'] == 0])

print(f"Final count - Relevance 1: {num_relevance_1_final}, Relevance 0: {num_relevance_0_final}")

    
# Save the complete climate finance data
merged_df.to_csv(wd + 'Data/train_set.csv', encoding='utf8', index=False, header=True)

print("Final balanced dataset saved")
