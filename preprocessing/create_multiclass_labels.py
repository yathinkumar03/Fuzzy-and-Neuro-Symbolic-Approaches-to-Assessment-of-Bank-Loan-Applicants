import pandas as pd
import numpy as np

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================
# LOAD MAPPED DATASETS
# ==========================================

credit_data = pd.read_csv(
    "../data/gmsc_mapped.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_mapped.csv"
)

# ==========================================
# SAMPLE DATASETS
# ==========================================

credit_data = credit_data.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

taiwan_data = taiwan_data.sample(
    n=10000,
    random_state=42
).reset_index(drop=True)

# ==========================================
# FUZZY VARIABLES
# ==========================================

financial_strength = ctrl.Antecedent(
    np.arange(-5000, 15000, 1),
    'financial_strength'
)

obligation_level = ctrl.Antecedent(
    np.arange(0, 10, 0.1),
    'obligation_level'
)

repayment_behavior = ctrl.Antecedent(
    np.arange(0, 20, 1),
    'repayment_behavior'
)

assessment = ctrl.Consequent(
    np.arange(0, 101, 1),
    'assessment'
)

# ==========================================
# MEMBERSHIP FUNCTIONS
# ==========================================

# ------------------------------------------
# FINANCIAL STRENGTH
# ------------------------------------------

financial_strength['low'] = fuzz.trimf(
    financial_strength.universe,
    [-5000, 0, 3000]
)

financial_strength['medium'] = fuzz.trimf(
    financial_strength.universe,
    [2000, 5000, 8000]
)

financial_strength['high'] = fuzz.trimf(
    financial_strength.universe,
    [7000, 15000, 15000]
)

# ------------------------------------------
# OBLIGATION LEVEL
# ------------------------------------------

obligation_level['low'] = fuzz.trimf(
    obligation_level.universe,
    [0, 0, 1]
)

obligation_level['medium'] = fuzz.trimf(
    obligation_level.universe,
    [0.5, 2, 4]
)

obligation_level['high'] = fuzz.trimf(
    obligation_level.universe,
    [3, 10, 10]
)

# ------------------------------------------
# REPAYMENT BEHAVIOR
# ------------------------------------------

repayment_behavior['good'] = fuzz.trimf(
    repayment_behavior.universe,
    [0, 0, 1]
)

repayment_behavior['average'] = fuzz.trimf(
    repayment_behavior.universe,
    [0, 3, 6]
)

repayment_behavior['bad'] = fuzz.trimf(
    repayment_behavior.universe,
    [5, 20, 20]
)

# ------------------------------------------
# ASSESSMENT
# ------------------------------------------

assessment['bad'] = fuzz.trimf(
    assessment.universe,
    [0, 0, 40]
)

assessment['fair'] = fuzz.trimf(
    assessment.universe,
    [30, 50, 70]
)

assessment['good'] = fuzz.trimf(
    assessment.universe,
    [60, 100, 100]
)

# ==========================================
# FUZZY RULES
# ==========================================

rule1 = ctrl.Rule(
    financial_strength['high']
    &
    obligation_level['low']
    &
    repayment_behavior['good'],
    assessment['good']
)

rule2 = ctrl.Rule(
    financial_strength['medium']
    &
    obligation_level['medium']
    &
    repayment_behavior['average'],
    assessment['fair']
)

rule3 = ctrl.Rule(
    financial_strength['low']
    &
    obligation_level['high']
    &
    repayment_behavior['bad'],
    assessment['bad']
)

rule4 = ctrl.Rule(
    repayment_behavior['bad'],
    assessment['bad']
)

rule5 = ctrl.Rule(
    financial_strength['high'],
    assessment['good']
)

rule6 = ctrl.Rule(
    obligation_level['high'],
    assessment['bad']
)

rule7 = ctrl.Rule(
    repayment_behavior['good']
    &
    obligation_level['low'],
    assessment['good']
)

rule8 = ctrl.Rule(
    financial_strength['low']
    &
    repayment_behavior['average'],
    assessment['fair']
)

# ==========================================
# CONTROL SYSTEM
# ==========================================

assessment_ctrl = ctrl.ControlSystem([
    rule1,
    rule2,
    rule3,
    rule4,
    rule5,
    rule6,
    rule7,
    rule8
])

# ==========================================
# MULTICLASS LABEL FUNCTION
# ==========================================

def generate_multiclass_labels(
    data,
    dataset_name
):

    scores = []

    # ======================================
    # GENERATE FUZZY SCORES
    # ======================================

    for i in range(len(data)):

        simulator = ctrl.ControlSystemSimulation(
            assessment_ctrl
        )

        simulator.input[
            'financial_strength'
        ] = data.loc[
            i,
            'financial_strength'
        ]

        simulator.input[
            'obligation_level'
        ] = data.loc[
            i,
            'obligation_level'
        ]

        simulator.input[
            'repayment_behavior'
        ] = data.loc[
            i,
            'repayment_behavior'
        ]

        simulator.compute()

        # ==================================
        # SAFE OUTPUT
        # ==================================

        if 'assessment' in simulator.output:

            score = simulator.output[
                'assessment'
            ]

        else:

            score = 50

        scores.append(score)

    # ======================================
    # PERCENTILE LABELING
    # ======================================

    scores_array = np.array(scores)

    low_threshold = np.percentile(
        scores_array,
        33
    )

    high_threshold = np.percentile(
        scores_array,
        66
    )

    labels = []

    for score in scores_array:

        if score <= low_threshold:

            category = "Bad"

        elif score <= high_threshold:

            category = "Fair"

        else:

            category = "Good"

        labels.append(category)

    # ======================================
    # STORE RESULTS
    # ======================================

    data['assessment_score'] = scores_array

    data['multiclass_label'] = labels

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print(f"\n========== {dataset_name} ==========\n")

    print("Low Threshold:")
    print(round(low_threshold, 2))

    print("\nHigh Threshold:")
    print(round(high_threshold, 2))

    print("\nClass Distribution:")

    print(
        data['multiclass_label']
        .value_counts()
    )

    print("\nSample Scores:")

    print(
        data[
            [
                'assessment_score',
                'multiclass_label'
            ]
        ].head(10)
    )

    return data

# ==========================================
# GENERATE MULTICLASS DATASETS
# ==========================================

credit_multiclass = generate_multiclass_labels(
    credit_data,
    "GIVE ME SOME CREDIT MULTICLASS"
)

taiwan_multiclass = generate_multiclass_labels(
    taiwan_data,
    "TAIWAN MULTICLASS"
)

# ==========================================
# SAVE DATASETS
# ==========================================

credit_multiclass.to_csv(
    "../data/gmsc_multiclass.csv",
    index=False
)

taiwan_multiclass.to_csv(
    "../data/taiwan_multiclass.csv",
    index=False
)

# ==========================================
# FINAL MESSAGE
# ==========================================

print(
    "\nMulticlass datasets created successfully!"
)