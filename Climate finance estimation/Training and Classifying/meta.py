import pandas as pd
import os
import sys
import json 

# Set the working directory.
# Paths resolve relative to this script's location: scripts live in
# 'Climate finance estimation/Training and Classifying/', so the 'Data/'
# folder is one level up. Override `wd` manually if you reorganize the tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
wd = os.path.abspath(os.path.join(_HERE, os.pardir, "Data")) + os.sep

def csv_import(name, delimiter=","):
    x = pd.read_csv(name, encoding='utf8', delimiter=delimiter,
                    dtype={'text': str,
                           "USD_Disbursement_Defl": float} )
    return x

df = csv_import(wd + "ClassifiedCRS.csv")

df = df.dropna(subset=['climate_class_number'])

length_start = df.shape[0]
if length_start < 2700000:
    print(length_start)

# Macro-categories, defined by class NAME (not by index) so that they stay correct
# whatever order the label dictionary has. Class names are those of the
# production model (Data/dictionary_classes.json, 17 classes).
ADAPTATION_CLASSES = {"Climate Adaptation", "Resilience"}
ENVIRONMENT_CLASSES = {
    "Environmental Policy Admin",
    "Forest Sustainability: Tropical, Sustainable Management, Deforestation, REDD+",
    "Enviro Ed Trainings", "Biodiv Conserv Prog", "Combat Desertif Convention",
    "National Capacities - Enviro Dev Plan Mainstreaming", "Wildlife conservation",
    "Marine-Coastal Protected Areas Mgmt. (CMB)",
}
MITIGATION_CLASSES = {
    "Wind power farms", "Geothermal Explr/Plants", "Renewable energy", "Solar PV Energy",
    "Hydro Power Plants Rehab", "Green Growth Strategies", "Air Pollution Mitigation",
}

# Translate the name-based groups into the class numbers used by the classifier,
# via the label dictionary written by multi-classifier.py.
with open(os.path.join(wd, "dictionary_classes.json")) as f:
    label_dict = json.load(f)
_unassigned = set(label_dict) - ADAPTATION_CLASSES - ENVIRONMENT_CLASSES - MITIGATION_CLASSES
if _unassigned:
    sys.exit(f"Classes without a macro-category: {sorted(_unassigned)}")
adaptation_categories = [label_dict[c] for c in ADAPTATION_CLASSES if c in label_dict]
environment_categories = [label_dict[c] for c in ENVIRONMENT_CLASSES if c in label_dict]
mitigation_categories = [label_dict[c] for c in MITIGATION_CLASSES if c in label_dict]


df['meta_category'] = 'None'

# Display rows with NaN in climate_class_number
df_na_climate_class = df[df['climate_class_number'].isna()]
print(f"Number of rows with NaN in climate_class_number: {df_na_climate_class.shape[0]}")

columns_to_keep = ['raw_text', 'climate_class_number', 'climate_relevance']  # Example variables
df_na_climate_class = df_na_climate_class[columns_to_keep].drop_duplicates()


# Group by 'raw_text' and check for mixed NaN and non-NaN in 'climate_class_number'
mixed_na = df.groupby('raw_text')['climate_class_number'].apply(lambda x: x.isna().any() and x.notna().any())

# Filter the original DataFrame to get rows where raw_text has mixed NaN and non-NaN climate_class_number values
df_mixed_na = df[df['raw_text'].isin(mixed_na[mixed_na].index)]

# Check for NaN values in climate_class_number
if df['climate_class_number'].isna().sum() > 0:
    print("Warning: NaN values found in climate_class_number. They will be set to 'None' in meta_category.")
    print("Rows with NaN in climate_class_number:", df[df['climate_class_number'].isna()].shape[0])


df.loc[df.climate_class_number.isin(adaptation_categories), 'meta_category'] = 'Adaptation'
print('Adaptation', df[df.climate_class_number.isin(adaptation_categories)].shape[0])
df.loc[df.climate_class_number.isin(mitigation_categories), 'meta_category'] = 'Mitigation'
print('Mitigation', df[df.climate_class_number.isin(mitigation_categories)].shape[0])
df.loc[df.climate_class_number.isin(environment_categories), 'meta_category'] = 'Environment'
print('Environment', df[df.climate_class_number.isin(environment_categories)].shape[0])

# Identify if any climate_class_number are not covered by the categories
unclassified = df[(df.meta_category == 'None') & (df.climate_relevance == 1)]
if not unclassified.empty:
    print("Unclassified categories detected:", unclassified['climate_class_number'].unique())

# Check plausibility
if df[df.meta_category == 'None'].shape[0] == df[df.climate_relevance == 0].shape[0]:
    print('Plausibility passed')
else:
    print('META SHAPE NONE: ', df[df.meta_category == 'None'].shape[0])
    print('Relevance SHAPE 0: ', df[df.climate_relevance == 0].shape[0])
    sys.exit()

length_end = df.shape[0]

if length_end == length_start:
    print("Second test passed")
else:
    print("Start Shape: ", length_start)
    print("End Shape: ", length_end)
    sys.exit()

df.to_csv(wd + 'climate_finance_total.csv', encoding='utf8', index=False, header=True, sep='|')
