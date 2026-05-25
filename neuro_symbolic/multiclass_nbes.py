import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASETS
# ==========================================

credit_data = pd.read_csv(
    "../data/gmsc_multiclass_encoded.csv"
)

taiwan_data = pd.read_csv(
    "../data/taiwan_multiclass_encoded.csv"
)

# ==========================================
# FEATURE SELECTION
# ==========================================

credit_features = [
    'age',
    'financial_strength',
    'obligation_level',
    'repayment_behavior'
]

taiwan_features = [
    'AGE',
    'financial_strength',
    'obligation_level',
    'repayment_behavior'
]

# ==========================================
# PREPARE DATA
# ==========================================

X_credit = credit_data[
    credit_features
].values

y_credit = credit_data[
    'encoded_label'
].values

X_taiwan = taiwan_data[
    taiwan_features
].values

y_taiwan = taiwan_data[
    'encoded_label'
].values

# ==========================================
# NORMALIZATION
# ==========================================

X_credit = X_credit / np.max(
    np.abs(X_credit),
    axis=0
)

X_taiwan = X_taiwan / np.max(
    np.abs(X_taiwan),
    axis=0
)

# ==========================================
# SOFTMAX FUNCTION
# ==========================================

def softmax(x):

    exp_x = np.exp(
        x - np.max(x)
    )

    return exp_x / np.sum(exp_x)

# ==========================================
# MULTICLASS NBES FUNCTION
# ==========================================

def train_multiclass_nbes(
    X,
    y,
    dataset_name
):

    print(f"\n========== TRAINING {dataset_name} ==========\n")

    # ======================================
    # PARAMETERS
    # ======================================

    num_features = X.shape[1]

    num_classes = 3

    learning_rate = 0.01

    epochs = 20

    # ======================================
    # INITIALIZE WEIGHTS
    # ======================================

    np.random.seed(42)

    weights = np.random.rand(
        num_features,
        num_classes
    )

    bias = np.random.rand(
        num_classes
    )

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

            logits = (
                np.dot(inputs, weights)
                + bias
            )

            probabilities = softmax(
                logits
            )

            # ==================================
            # ONE-HOT TARGET
            # ==================================

            target = np.zeros(
                num_classes
            )

            target[
                int(y[i])
            ] = 1

            # ==================================
            # ERROR
            # ==================================

            error = (
                target
                - probabilities
            )

            total_error += np.sum(
                np.abs(error)
            )

            # ==================================
            # WEIGHT UPDATE
            # ==================================

            weights += (
                learning_rate
                *
                np.outer(
                    inputs,
                    error
                )
            )

            bias += (
                learning_rate
                * error
            )

        # ======================================
        # DISPLAY EPOCH INFO
        # ======================================

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Total Error: "
            f"{round(total_error, 2)}"
        )

    # ======================================
    # FINAL PREDICTIONS
    # ======================================

    predictions = []

    for i in range(len(X)):

        logits = (
            np.dot(X[i], weights)
            + bias
        )

        probabilities = softmax(
            logits
        )

        prediction = np.argmax(
            probabilities
        )

        predictions.append(
            prediction
        )

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

    print(
        f"\n========== "
        f"{dataset_name} RESULTS ==========\n"
    )

    print("Accuracy:")

    print(
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(report)

    return cm

# ==========================================
# RUN GIVE ME SOME CREDIT NBES
# ==========================================

credit_cm = train_multiclass_nbes(
    X_credit,
    y_credit,
    "GIVE ME SOME CREDIT MULTICLASS NBES"
)

# ==========================================
# RUN TAIWAN NBES
# ==========================================

taiwan_cm = train_multiclass_nbes(
    X_taiwan,
    y_taiwan,
    "TAIWAN MULTICLASS NBES"
)