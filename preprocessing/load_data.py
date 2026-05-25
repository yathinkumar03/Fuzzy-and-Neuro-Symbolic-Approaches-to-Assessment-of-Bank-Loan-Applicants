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
# DISPLAY GIVE ME SOME CREDIT DATASET
# ==========================================

print("\n========== GIVE ME SOME CREDIT DATASET ==========\n")

print(credit_data.head())

print("\nDataset Shape:")
print(credit_data.shape)

print("\nDataset Columns:")
print(credit_data.columns)

print("\nMissing Values:")
print(credit_data.isnull().sum())

# ==========================================
# DISPLAY TAIWAN DATASET
# ==========================================

print("\n========== TAIWAN DATASET ==========\n")

print(taiwan_data.head())

print("\nDataset Shape:")
print(taiwan_data.shape)

print("\nDataset Columns:")
print(taiwan_data.columns)

print("\nMissing Values:")
print(taiwan_data.isnull().sum())