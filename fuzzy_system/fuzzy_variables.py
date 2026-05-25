import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ==========================================
# GENERIC FUZZY VARIABLES
# ==========================================

# Applicant Age
age = ctrl.Antecedent(
    np.arange(18, 81, 1),
    'age'
)

# Financial Strength
financial_strength = ctrl.Antecedent(
    np.arange(0, 100001, 1),
    'financial_strength'
)

# Obligation Level
obligation_level = ctrl.Antecedent(
    np.arange(0, 100001, 1),
    'obligation_level'
)

# Final Applicant Assessment
applicant_assessment = ctrl.Consequent(
    np.arange(0, 101, 1),
    'applicant_assessment'
)

# ==========================================
# AGE MEMBERSHIP FUNCTIONS
# ==========================================

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

# ==========================================
# FINANCIAL STRENGTH MEMBERSHIP FUNCTIONS
# ==========================================

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

# ==========================================
# OBLIGATION LEVEL MEMBERSHIP FUNCTIONS
# ==========================================

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

# ==========================================
# APPLICANT ASSESSMENT MEMBERSHIP FUNCTIONS
# ==========================================

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
# DISPLAY MEMBERSHIP FUNCTIONS
# ==========================================

print("\nDisplaying fuzzy membership functions...\n")

age.view()

financial_strength.view()

obligation_level.view()

applicant_assessment.view()

plt.show()