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
# LOAD TAIWAN DATASET
# ==========================================

data = pd.read_csv(
    "../data/taiwan_mapped.csv"
)

# ==========================================
# FUZZY VARIABLES
# ==========================================

age = ctrl.Antecedent(
    np.arange(18, 81, 1),
    'age'
)

financial_strength = ctrl.Antecedent(
    np.arange(0, 1000001, 1),
    'financial_strength'
)

obligation_level = ctrl.Antecedent(
    np.arange(-200000, 1000001, 1),
    'obligation_level'
)

repayment_behavior = ctrl.Antecedent(
    np.arange(-2, 10, 1),
    'repayment_behavior'
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
    [18, 18, 30]
)

age['middle'] = fuzz.trimf(
    age.universe,
    [25, 40, 55]
)

age['old'] = fuzz.trimf(
    age.universe,
    [50, 80, 80]
)

# Financial Strength
financial_strength['low'] = fuzz.trimf(
    financial_strength.universe,
    [0, 0, 150000]
)

financial_strength['medium'] = fuzz.trimf(
    financial_strength.universe,
    [100000, 300000, 500000]
)

financial_strength['high'] = fuzz.trimf(
    financial_strength.universe,
    [400000, 1000000, 1000000]
)

# Obligation Level
obligation_level['low'] = fuzz.trimf(
    obligation_level.universe,
    [-200000, 0, 100000]
)

obligation_level['medium'] = fuzz.trimf(
    obligation_level.universe,
    [50000, 250000, 500000]
)

obligation_level['high'] = fuzz.trimf(
    obligation_level.universe,
    [400000, 1000000, 1000000]
)

# Repayment Behavior
repayment_behavior['good'] = fuzz.trimf(
    repayment_behavior.universe,
    [-2, -2, 0]
)

repayment_behavior['moderate'] = fuzz.trimf(
    repayment_behavior.universe,
    [0, 2, 4]
)

repayment_behavior['poor'] = fuzz.trimf(
    repayment_behavior.universe,
    [3, 9, 9]
)

# Output
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
# TAIWAN FUZZY RULES
# ==========================================

rule1 = ctrl.Rule(
    repayment_behavior['good'] &
    financial_strength['high'],
    applicant_assessment['good']
)

rule2 = ctrl.Rule(
    repayment_behavior['poor'],
    applicant_assessment['bad']
)

rule3 = ctrl.Rule(
    obligation_level['high'] &
    repayment_behavior['poor'],
    applicant_assessment['bad']
)

rule4 = ctrl.Rule(
    financial_strength['low'] &
    obligation_level['high'],
    applicant_assessment['bad']
)

rule5 = ctrl.Rule(
    repayment_behavior['moderate'] &
    financial_strength['medium'],
    applicant_assessment['fair']
)

rule6 = ctrl.Rule(
    age['young'] &
    repayment_behavior['good'],
    applicant_assessment['good']
)

rule7 = ctrl.Rule(
    age['old'] &
    repayment_behavior['poor'],
    applicant_assessment['bad']
)

rule8 = ctrl.Rule(
    financial_strength['high'] &
    obligation_level['low'],
    applicant_assessment['good']
)

rule9 = ctrl.Rule(
    obligation_level['high'],
    applicant_assessment['bad']
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
# EVALUATION
# ==========================================

y_true = []
y_pred = []

print("\n========== TAIWAN FES EVALUATION ==========\n")

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

    simulator.input[
        'repayment_behavior'
    ] = data.loc[i, 'repayment_behavior']

    simulator.compute()

    if 'applicant_assessment' in simulator.output:

        score = simulator.output[
            'applicant_assessment'
        ]

    else:
        score = 50

    if score >= 60:
        prediction = 1

    else:
        prediction = 0

    y_pred.append(prediction)

    y_true.append(data.loc[i, 'target'])

# ==========================================
# METRICS
# ==========================================

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

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)