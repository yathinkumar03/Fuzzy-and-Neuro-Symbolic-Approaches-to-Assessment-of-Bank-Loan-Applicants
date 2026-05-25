# Taiwan Creditworthiness Assessment using Fuzzy Expert System and Neuro-Rule Based Expert System

## Overview

This project presents a hybrid Artificial Intelligence framework for creditworthiness assessment using a Fuzzy Expert System (FES) and a Neuro-Rule Based Expert System (NBES). The system evaluates applicant financial risk using the Taiwan Credit Card Default Dataset and combines interpretable fuzzy reasoning with adaptive neural learning.

The project demonstrates how neuro-symbolic AI can be applied to financial risk analysis and multiclass creditworthiness prediction.

---

## Objectives

- Develop an interpretable credit assessment system
- Model expert financial reasoning using fuzzy logic
- Perform multiclass applicant classification
- Integrate fuzzy reasoning with neural learning
- Improve prediction accuracy while maintaining explainability

---

## Dataset

### Taiwan Credit Card Default Dataset

The dataset contains financial and repayment information of credit card clients including:

- Credit limit
- Bill amounts
- Repayment history
- Demographic information
- Default payment status

---

## Feature Engineering

The following financial indicators were generated from raw banking attributes:

### Financial Strength (FS)

```math
FS = LIMIT\_BAL - BILL\_AMT1
```

Represents remaining financial capability.

---

### Obligation Level (OL)

```math
OL = \frac{BILL\_AMT1}{LIMIT\_BAL + 1}
```

Represents debt burden ratio.

---

### Repayment Behavior (RB)

```math
RB = PAY\_0 + PAY\_2 + PAY\_3
```

Represents repayment discipline using historical delay information.

---

## Fuzzy Expert System (FES)

The Fuzzy Expert System models human-like financial reasoning using:

- Triangular membership functions
- Linguistic variables
- IF-THEN fuzzy rules
- Fuzzy inference
- Centroid defuzzification

### Input Variables

- Financial Strength
- Obligation Level
- Repayment Behavior

### Output

- Assessment Score
- Multiclass Labels:
  - Bad
  - Fair
  - Good

---

## Neuro-Rule Based Expert System (NBES)

The NBES learns from fuzzy-generated multiclass labels using neural learning.

### Components

- Weighted feature learning
- Softmax multiclass classification
- Weight and bias updates
- Error minimization

### Input Feature Vector

```math
X = [FS, OL, RB, Age]
```

### Weighted Sum

```math
z = \sum w_i x_i + b
```

### Softmax Function

```math
P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

---

## Evaluation Metrics

The system is evaluated using:

- Accuracy
- Sensitivity
- Specificity
- Confusion Matrix
- Classification Report

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Scikit-fuzzy
- Matplotlib

---

## Project Structure

```text
BankLoanAssessment/
│
├── data/
├── preprocessing/
├── fuzzy_system/
├── neuro_symbolic/
├── results/
├── visualizations/
├── README.md
└── requirements.txt
```

---

## Results

The proposed neuro-symbolic framework achieved strong multiclass classification performance while maintaining transparent and interpretable reasoning.

The project demonstrates that combining fuzzy expert reasoning with neural learning improves financial risk assessment effectiveness.

---

## Future Scope

- Deep neuro-fuzzy architectures
- Real-time banking deployment
- Explainable AI dashboards
- Hybrid reinforcement learning integration
- Large-scale financial analytics

---

## Author

Yathin Kumar

B.Tech Artificial Intelligence

Amrita Vishwa Vidyapeetham, Bengaluru

---

## License

This project is developed for academic and research purposes.
