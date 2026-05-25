import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASETS
# ==========================================

german_data = pd.read_csv(
    "../data/german_mapped.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_mapped.csv"
)

# ==========================================
# FEATURE SELECTION
# ==========================================

german_features = [
    'age',
    'financial_strength',
    'obligation_level'
]

taiwan_features = [
    'age',
    'financial_strength',
    'obligation_level',
    'repayment_behavior'
]

# ==========================================
# PREPARE DATA
# ==========================================

X_german = german_data[
    german_features
].values

y_german = german_data[
    'target'
].values

X_taiwan = taiwan_data[
    taiwan_features
].values

y_taiwan = taiwan_data[
    'target'
].values

# ==========================================
# NORMALIZATION
# ==========================================

X_german = X_german / np.max(
    X_german,
    axis=0
)

X_taiwan = X_taiwan / np.max(
    X_taiwan,
    axis=0
)

# ==========================================
# SIGMOID ACTIVATION
# ==========================================

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_neurule_system(
    X,
    y,
    weights,
    bias,
    learning_rate,
    epochs,
    dataset_name
):

    print(f"\n========== TRAINING {dataset_name} ==========\n")

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(epochs):

        total_error = 0

        for i in range(len(X)):

            inputs = X[i]

            # ==================================
            # FORWARD PASS
            # ==================================

            weighted_sum = (
                np.dot(inputs, weights)
                + bias
            )

            prediction = sigmoid(
                weighted_sum
            )

            # ==================================
            # ERROR
            # ==================================

            error = y[i] - prediction

            total_error += abs(error)

            # ==================================
            # GRADIENT UPDATE
            # ==================================

            weights += (
                learning_rate
                * error
                * prediction
                * (1 - prediction)
                * inputs
            )

            # Bias update
            bias += (
                learning_rate
                * error
                * prediction
                * (1 - prediction)
            )

        # ======================================
        # EPOCH DISPLAY
        # ======================================

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Total Error: {round(total_error, 2)}"
        )

    # ======================================
    # FINAL PREDICTIONS
    # ======================================

    predictions = []

    for i in range(len(X)):

        weighted_sum = (
            np.dot(X[i], weights)
            + bias
        )

        probability = sigmoid(
            weighted_sum
        )

        # Binary classification
        if probability >= 0.5:
            prediction = 1

        else:
            prediction = 0

        predictions.append(prediction)

    # ======================================
    # EVALUATION
    # ======================================

    accuracy = accuracy_score(
        y,
        predictions
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    report = classification_report(
        y,
        predictions
    )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print(f"\n========== {dataset_name} RESULTS ==========\n")

    print("Final Weights:")
    print(weights)

    print("\nFinal Bias:")
    print(round(bias, 4))

    print("\nAccuracy:")
    print(round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    return weights, bias

# ==========================================
# INITIAL PARAMETERS
# ==========================================

np.random.seed(42)

german_weights = np.random.rand(3)

taiwan_weights = np.random.rand(4)

german_bias = np.random.rand()

taiwan_bias = np.random.rand()

learning_rate = 0.01

epochs = 20

# ==========================================
# TRAIN GERMAN NBES
# ==========================================

german_weights, german_bias = train_neurule_system(
    X_german,
    y_german,
    german_weights,
    german_bias,
    learning_rate,
    epochs,
    "GERMAN NBES"
)

# ==========================================
# TRAIN TAIWAN NBES
# ==========================================

taiwan_weights, taiwan_bias = train_neurule_system(
    X_taiwan,
    y_taiwan,
    taiwan_weights,
    taiwan_bias,
    learning_rate,
    epochs,
    "TAIWAN NBES"
)