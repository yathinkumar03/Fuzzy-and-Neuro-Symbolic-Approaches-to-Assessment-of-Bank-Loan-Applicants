import pandas as pd

# ==========================================
# LOAD MULTICLASS DATASETS
# ==========================================

credit_data = pd.read_csv(
    "../data/gmsc_multiclass.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_multiclass.csv"
)

# ==========================================
# LABEL ENCODING
# ==========================================

label_mapping = {
    "Bad": 0,
    "Fair": 1,
    "Good": 2
}

# ==========================================
# ENCODE GIVE ME SOME CREDIT
# ==========================================

credit_data[
    'encoded_label'
] = credit_data[
    'multiclass_label'
].map(label_mapping)

# ==========================================
# ENCODE TAIWAN
# ==========================================

taiwan_data[
    'encoded_label'
] = taiwan_data[
    'multiclass_label'
].map(label_mapping)

# ==========================================
# DISPLAY GIVE ME SOME CREDIT DATASET
# ==========================================

print("\n========== GIVE ME SOME CREDIT ==========\n")

print(
    credit_data[
        [
            'assessment_score',
            'multiclass_label',
            'encoded_label'
        ]
    ].head(10)
)

print("\nClass Distribution:")

print(
    credit_data[
        'multiclass_label'
    ].value_counts()
)

# ==========================================
# DISPLAY TAIWAN DATASET
# ==========================================

print("\n========== TAIWAN DATASET ==========\n")

print(
    taiwan_data[
        [
            'assessment_score',
            'multiclass_label',
            'encoded_label'
        ]
    ].head(10)
)

print("\nClass Distribution:")

print(
    taiwan_data[
        'multiclass_label'
    ].value_counts()
)

# ==========================================
# SAVE DATASETS
# ==========================================

credit_data.to_csv(
    "../data/gmsc_multiclass_encoded.csv",
    index=False
)

taiwan_data.to_csv(
    "../data/taiwan_multiclass_encoded.csv",
    index=False
)

print("\nEncoded multiclass datasets saved successfully!")