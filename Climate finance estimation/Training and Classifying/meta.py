import pandas as pd
import os
import sys 

# Set the working directory
wd = ".../Climate finance estimation/Data/"
os.chdir(wd)

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

# Add meta categories
adaptation_categories = [10, 13]
environment_categories = [0, 1, 2, 5, 9, 12, 14, 15]
mitigation_categories = [3, 4, 6, 7, 8, 11, 16]


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
