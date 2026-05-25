import pandas as pd

# ==========================================
# LOAD GIVE ME SOME CREDIT DATASET
# ==========================================

credit_data = pd.read_csv(
    "../data/give_me_some_credit.csv"
)

# ==========================================
# LOAD TAIWAN DATASET
# ==========================================

taiwan_data = pd.read_csv(
    "../data/taiwan_credit.csv"
)

# ==========================================
# REMOVE UNNECESSARY COLUMNS
# ==========================================

credit_data.drop(
    columns=['Unnamed: 0'],
    inplace=True
)

taiwan_data.drop(
    columns=['ID'],
    inplace=True
)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

# Monthly income
credit_data[
    'MonthlyIncome'
] = credit_data[
    'MonthlyIncome'
].fillna(
    credit_data[
        'MonthlyIncome'
    ].median()
)

# Dependents
credit_data[
    'NumberOfDependents'
] = credit_data[
    'NumberOfDependents'
].fillna(
    credit_data[
        'NumberOfDependents'
    ].median()
)

# ==========================================
# DISPLAY CLEANED DATASET
# ==========================================

print("\n========== CLEANED GIVE ME SOME CREDIT DATASET ==========\n")

print(credit_data.head())

print("\nDataset Shape:")
print(credit_data.shape)

print("\nMissing Values:")
print(credit_data.isnull().sum())

# ==========================================
# DISPLAY TAIWAN DATASET
# ==========================================

print("\n========== CLEANED TAIWAN DATASET ==========\n")

print(taiwan_data.head())

print("\nDataset Shape:")
print(taiwan_data.shape)

print("\nMissing Values:")
print(taiwan_data.isnull().sum())

# ==========================================
# SAVE CLEANED DATASETS
# ==========================================

credit_data.to_csv(
    "../data/gmsc_cleaned.csv",
    index=False
)

taiwan_data.to_csv(
    "../data/taiwan_credit_cleaned.csv",
    index=False
)

print("\nCleaned datasets saved successfully!")