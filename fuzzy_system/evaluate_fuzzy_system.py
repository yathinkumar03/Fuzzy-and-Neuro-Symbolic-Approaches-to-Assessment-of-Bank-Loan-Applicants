import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD MAPPED DATASETS
# ==========================================

german_data = pd.read_csv(
    "../data/german_mapped.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_mapped.csv"
)

# ==========================================
# GENERIC FUZZY VARIABLES
# ==========================================

age = ctrl.Antecedent(
    np.arange(18, 81, 1),
    'age'
)

financial_strength = ctrl.Antecedent(
    np.arange(0, 100001, 1),
    'financial_strength'
)

obligation_level = ctrl.Antecedent(
    np.arange(0, 100001, 1),
    'obligation_level'
)

applicant_assessment = ctrl.Consequent(
    np.arange(0, 101, 1),
    'applicant_assessment'
)

# ==========================================
# MEMBERSHIP FUNCTIONS
# ==========================================

# Age
age['young'] = fuzz.trimf(
    age.universe,
    [18, 18, 35]
)

age['normal'] = fuzz.trimf(
    age.universe,
    [30, 45, 60]
)

age['old'] = fuzz.trimf(
    age.universe,
    [55, 80, 80]
)

# Financial Strength
financial_strength['low'] = fuzz.trimf(
    financial_strength.universe,
    [0, 0, 30000]
)

financial_strength['medium'] = fuzz.trimf(
    financial_strength.universe,
    [20000, 50000, 80000]
)

financial_strength['high'] = fuzz.trimf(
    financial_strength.universe,
    [70000, 100000, 100000]
)

# Obligation Level
obligation_level['low'] = fuzz.trimf(
    obligation_level.universe,
    [0, 0, 30000]
)

obligation_level['medium'] = fuzz.trimf(
    obligation_level.universe,
    [20000, 50000, 80000]
)

obligation_level['high'] = fuzz.trimf(
    obligation_level.universe,
    [70000, 100000, 100000]
)

# Applicant Assessment
applicant_assessment['bad'] = fuzz.trimf(
    applicant_assessment.universe,
    [0, 0, 40]
)

applicant_assessment['fair'] = fuzz.trimf(
    applicant_assessment.universe,
    [30, 50, 70]
)

applicant_assessment['good'] = fuzz.trimf(
    applicant_assessment.universe,
    [60, 100, 100]
)

# ==========================================
# FUZZY RULES
# ==========================================

rule1 = ctrl.Rule(
    financial_strength['high'] &
    obligation_level['low'],
    applicant_assessment['good']
)

rule2 = ctrl.Rule(
    financial_strength['low'] &
    obligation_level['high'],
    applicant_assessment['bad']
)

rule3 = ctrl.Rule(
    financial_strength['medium'] &
    obligation_level['medium'],
    applicant_assessment['fair']
)

rule4 = ctrl.Rule(
    age['young'] &
    financial_strength['high'],
    applicant_assessment['good']
)

rule5 = ctrl.Rule(
    age['old'] &
    obligation_level['high'],
    applicant_assessment['bad']
)

rule6 = ctrl.Rule(
    obligation_level['high'],
    applicant_assessment['bad']
)

rule7 = ctrl.Rule(
    financial_strength['high'],
    applicant_assessment['good']
)

rule8 = ctrl.Rule(
    financial_strength['low'],
    applicant_assessment['bad']
)

rule9 = ctrl.Rule(
    age['normal'] &
    financial_strength['medium'],
    applicant_assessment['fair']
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
    rule8,
    rule9
])

# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_dataset(data, dataset_name):

    print(f"\n========== {dataset_name} ==========\n")

    y_true = []
    y_pred = []

    for i in range(len(data)):

        simulator = ctrl.ControlSystemSimulation(
            assessment_ctrl
        )

        simulator.input['age'] = data.loc[i, 'age']

        simulator.input[
            'financial_strength'
        ] = data.loc[i, 'financial_strength']

        simulator.input[
            'obligation_level'
        ] = data.loc[i, 'obligation_level']

        # Run inference
        simulator.compute()

        # Safe handling
        if 'applicant_assessment' in simulator.output:

            score = simulator.output[
                'applicant_assessment'
            ]

        else:
            score = 50

        # Convert score to binary prediction
        if score >= 60:
            prediction = 1

        else:
            prediction = 0

        y_pred.append(prediction)

        y_true.append(data.loc[i, 'target'])

    # ======================================
    # METRICS
    # ======================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred
    )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("Accuracy:")
    print(round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

# ==========================================
# RUN BOTH DATASET EVALUATIONS
# ==========================================

evaluate_dataset(
    german_data,
    "GERMAN DATASET EVALUATION"
)

evaluate_dataset(
    taiwan_data,
    "TAIWAN DATASET EVALUATION"
)