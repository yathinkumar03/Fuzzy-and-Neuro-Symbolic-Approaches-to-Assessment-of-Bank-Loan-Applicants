import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

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
# GENERALIZED FUZZY RULES
# ==========================================

# Strong financial condition
rule1 = ctrl.Rule(
    financial_strength['high'] &
    obligation_level['low'],
    applicant_assessment['good']
)

# Weak financial condition
rule2 = ctrl.Rule(
    financial_strength['low'] &
    obligation_level['high'],
    applicant_assessment['bad']
)

# Moderate condition
rule3 = ctrl.Rule(
    financial_strength['medium'] &
    obligation_level['medium'],
    applicant_assessment['fair']
)

# Young + good finances
rule4 = ctrl.Rule(
    age['young'] &
    financial_strength['high'],
    applicant_assessment['good']
)

# Old + high obligations
rule5 = ctrl.Rule(
    age['old'] &
    obligation_level['high'],
    applicant_assessment['bad']
)

# High obligations generally risky
rule6 = ctrl.Rule(
    obligation_level['high'],
    applicant_assessment['bad']
)

# Strong finances generally good
rule7 = ctrl.Rule(
    financial_strength['high'],
    applicant_assessment['good']
)

# Weak finances risky
rule8 = ctrl.Rule(
    financial_strength['low'],
    applicant_assessment['bad']
)

# Normal age + medium finance
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

assessment_simulator = ctrl.ControlSystemSimulation(
    assessment_ctrl
)

# ==========================================
# SAMPLE TEST
# ==========================================

assessment_simulator.input['age'] = 30

assessment_simulator.input['financial_strength'] = 70000

assessment_simulator.input['obligation_level'] = 20000

# ==========================================
# RUN FUZZY INFERENCE
# ==========================================

assessment_simulator.compute()

# ==========================================
# DISPLAY OUTPUT
# ==========================================

print("\nApplicant Assessment Score:")

# Safe output handling

if 'applicant_assessment' in assessment_simulator.output:

    score = assessment_simulator.output[
        'applicant_assessment'
    ]

else:
    score = 50

print(round(score, 2))

applicant_assessment.view(
    sim=assessment_simulator
)

plt.show()