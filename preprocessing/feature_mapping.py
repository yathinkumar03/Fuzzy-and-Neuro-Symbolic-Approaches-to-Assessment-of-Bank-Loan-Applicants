import pandas as pd

# ==========================================
# LOAD CLEANED DATASETS
# ==========================================

credit_data = pd.read_csv(
    "../data/gmsc_cleaned.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_credit_cleaned.csv"
)

# ==========================================
# FEATURE MAPPING
# ==========================================

# ==========================================
# GIVE ME SOME CREDIT DATASET
# ==========================================

# ------------------------------------------
# FINANCIAL STRENGTH
# ------------------------------------------

credit_data[
    'financial_strength'
] = (
    credit_data['MonthlyIncome']
    *
    (
        1 -
        credit_data[
            'RevolvingUtilizationOfUnsecuredLines'
        ]
    )
)

# ------------------------------------------
# OBLIGATION LEVEL
# ------------------------------------------

credit_data[
    'obligation_level'
] = (
    credit_data['DebtRatio']
    *
    (
        credit_data[
            'NumberOfDependents'
        ] + 1
    )
)

# ------------------------------------------
# REPAYMENT BEHAVIOR
# ------------------------------------------

credit_data[
    'repayment_behavior'
] = (
    credit_data[
        'NumberOfTimes90DaysLate'
    ]
    +
    credit_data[
        'NumberOfTime30-59DaysPastDueNotWorse'
    ]
    +
    credit_data[
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
)

# ==========================================
# OUTLIER HANDLING - GIVE ME SOME CREDIT
# ==========================================

credit_data['financial_strength'] = (
    credit_data['financial_strength']
    .clip(
        lower=
        credit_data[
            'financial_strength'
        ].quantile(0.01),

        upper=
        credit_data[
            'financial_strength'
        ].quantile(0.99)
    )
)

credit_data['obligation_level'] = (
    credit_data['obligation_level']
    .clip(
        lower=
        credit_data[
            'obligation_level'
        ].quantile(0.01),

        upper=
        credit_data[
            'obligation_level'
        ].quantile(0.99)
    )
)

credit_data['repayment_behavior'] = (
    credit_data['repayment_behavior']
    .clip(
        lower=
        credit_data[
            'repayment_behavior'
        ].quantile(0.01),

        upper=
        credit_data[
            'repayment_behavior'
        ].quantile(0.99)
    )
)

# ==========================================
# TARGET VARIABLE
# ==========================================

credit_data[
    'target'
] = credit_data[
    'SeriousDlqin2yrs'
]

# ==========================================
# TAIWAN DATASET
# ==========================================

# ------------------------------------------
# FINANCIAL STRENGTH
# ------------------------------------------

taiwan_data[
    'financial_strength'
] = (
    taiwan_data['LIMIT_BAL']
    -
    taiwan_data['BILL_AMT1']
)

# ------------------------------------------
# OBLIGATION LEVEL
# ------------------------------------------

taiwan_data[
    'obligation_level'
] = (
    taiwan_data['BILL_AMT1']
    /
    (
        taiwan_data['LIMIT_BAL']
        + 1
    )
)

# ------------------------------------------
# REPAYMENT BEHAVIOR
# ------------------------------------------

taiwan_data[
    'repayment_behavior'
] = (
    taiwan_data['PAY_0']
    +
    taiwan_data['PAY_2']
    +
    taiwan_data['PAY_3']
)

# ==========================================
# OUTLIER HANDLING - TAIWAN
# ==========================================

taiwan_data['financial_strength'] = (
    taiwan_data['financial_strength']
    .clip(
        lower=
        taiwan_data[
            'financial_strength'
        ].quantile(0.01),

        upper=
        taiwan_data[
            'financial_strength'
        ].quantile(0.99)
    )
)

taiwan_data['obligation_level'] = (
    taiwan_data['obligation_level']
    .clip(
        lower=
        taiwan_data[
            'obligation_level'
        ].quantile(0.01),

        upper=
        taiwan_data[
            'obligation_level'
        ].quantile(0.99)
    )
)

taiwan_data['repayment_behavior'] = (
    taiwan_data['repayment_behavior']
    .clip(
        lower=
        taiwan_data[
            'repayment_behavior'
        ].quantile(0.01),

        upper=
        taiwan_data[
            'repayment_behavior'
        ].quantile(0.99)
    )
)

# ==========================================
# TARGET VARIABLE
# ==========================================

taiwan_data[
    'target'
] = taiwan_data[
    'default.payment.next.month'
]

# ==========================================
# DISPLAY GIVE ME SOME CREDIT DATASET
# ==========================================

print(
    "\n========== GIVE ME SOME CREDIT "
    "MAPPED DATASET ==========\n"
)

print(
    credit_data[
        [
            'financial_strength',
            'obligation_level',
            'repayment_behavior',
            'target'
        ]
    ].head()
)

# ==========================================
# DISPLAY TAIWAN DATASET
# ==========================================

print(
    "\n========== TAIWAN MAPPED "
    "DATASET ==========\n"
)

print(
    taiwan_data[
        [
            'financial_strength',
            'obligation_level',
            'repayment_behavior',
            'target'
        ]
    ].head()
)

# ==========================================
# DISPLAY STATISTICS
# ==========================================

print(
    "\n========== GIVE ME SOME CREDIT "
    "STATISTICS ==========\n"
)

print(
    credit_data[
        [
            'financial_strength',
            'obligation_level',
            'repayment_behavior'
        ]
    ].describe()
)

# ==========================================

print(
    "\n========== TAIWAN STATISTICS ==========\n"
)

print(
    taiwan_data[
        [
            'financial_strength',
            'obligation_level',
            'repayment_behavior'
        ]
    ].describe()
)

# ==========================================
# SAVE DATASETS
# ==========================================

credit_data.to_csv(
    "../data/gmsc_mapped.csv",
    index=False
)

taiwan_data.to_csv(
    "../data/taiwan_mapped.csv",
    index=False
)

# ==========================================
# FINAL MESSAGE
# ==========================================

print(
    "\nMapped datasets saved successfully!"
)