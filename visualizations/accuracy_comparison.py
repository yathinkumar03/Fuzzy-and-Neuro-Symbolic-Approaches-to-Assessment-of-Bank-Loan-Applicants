import pandas as pd

# ==========================================
# ENTER FES RESULTS
# ==========================================

fes_accuracy = [
    0.90,
    0.73,
    0.86
]

fes_sensitivity = [
    0.80,
    0.60,
    0.80
]

fes_specificity = [
    0.95,
    0.80,
    0.85
]

# ==========================================
# ENTER NBES RESULTS
# ==========================================

nbes_accuracy = [
    0.83,
    0.76,
    0.86
]

nbes_sensitivity = [
    0.87,
    0.60,
    0.83
]

nbes_specificity = [
    0.81,
    0.85,
    0.88
]

# ==========================================
# CALCULATE AVERAGES
# ==========================================

avg_fes_accuracy = round(
    sum(fes_accuracy) / 3,
    2
)

avg_nbes_accuracy = round(
    sum(nbes_accuracy) / 3,
    2
)

avg_fes_sensitivity = round(
    sum(fes_sensitivity) / 3,
    2
)

avg_nbes_sensitivity = round(
    sum(nbes_sensitivity) / 3,
    2
)

avg_fes_specificity = round(
    sum(fes_specificity) / 3,
    2
)

avg_nbes_specificity = round(
    sum(nbes_specificity) / 3,
    2
)

# ==========================================
# CREATE TABLE
# ==========================================

results_table = pd.DataFrame({

    'Applicant Assessment': [
        'Bad',
        'Fair',
        'Good',
        'Average'
    ],

    'Accuracy FES': [
        fes_accuracy[0],
        fes_accuracy[1],
        fes_accuracy[2],
        avg_fes_accuracy
    ],

    'Accuracy NBES': [
        nbes_accuracy[0],
        nbes_accuracy[1],
        nbes_accuracy[2],
        avg_nbes_accuracy
    ],

    'Sensitivity FES': [
        fes_sensitivity[0],
        fes_sensitivity[1],
        fes_sensitivity[2],
        avg_fes_sensitivity
    ],

    'Sensitivity NBES': [
        nbes_sensitivity[0],
        nbes_sensitivity[1],
        nbes_sensitivity[2],
        avg_nbes_sensitivity
    ],

    'Specificity FES': [
        fes_specificity[0],
        fes_specificity[1],
        fes_specificity[2],
        avg_fes_specificity
    ],

    'Specificity NBES': [
        nbes_specificity[0],
        nbes_specificity[1],
        nbes_specificity[2],
        avg_nbes_specificity
    ]

})

# ==========================================
# DISPLAY TABLE
# ==========================================

print(
    "\n========== EVALUATION RESULTS ==========\n"
)

print(results_table)

# ==========================================
# SAVE TABLE
# ==========================================

results_table.to_csv(
    "../results/evaluation_results_table.csv",
    index=False
)

print(
    "\nEvaluation table saved successfully!"
)