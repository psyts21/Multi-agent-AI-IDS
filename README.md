# 🛡️ Multi-Agent Intrusion Detection System (IDS)

This project explores whether distributing intelligence across multiple specialised agents can improve intrusion detection performance. It compares a modular **multi-agent IDS architecture** with three  **single-agent models** using the NSL-KDD dataset.


##  Models Implemented

### 🔹 Single-Agent models
Each model was trained on the entire dataset to classify all five classes:
- `RandomForestIDS`: Based on `sklearn.ensemble.RandomForestClassifier`
- `XGBoostIDS`: Based on `xgboost.XGBClassifier`
- `VotingClassifierIDS`: A hard voting ensemble combining the two above

### 🔸 Multi-Agent IDS
A modular architecture composed of:
- `DoSAgent`: XGBoost (binary classifier for DoS detection)
- `NormalTrafficAgent`: XGBoost
- `ProbeAgent`: MLP (for subtle, non-linear Probe patterns)
- `R2LAgent`: MLP
- `U2RAgent`: MLP
- `CoordinatorAgent`: Aggregates predictions via **weighted fusion**, where each agent's influence is inversely proportional to class frequency × recall.

## 🛠️ Technologies Used

- Python 3.x
- `scikit-learn`
- `xgboost`
- `imblearn` (SMOTE, SMOTETomek)
- `pandas`, `numpy`, `matplotlib`

## 🧪 Evaluation

All models were evaluated over 3 experimental runs. Metrics included:
- Precision, Recall, F1-Score
- False Positive Rate
- False Negative Rate
- Mean and Standard Deviation

## 🧠 System Features

- **SMOTETomek** balancing to handle rare attack classes
- **RandomizedSearchCV** for hyperparameter tuning
- **Reusable agent-based code structure** using OOP principles
- **Modular Coordinator** for easy expansion and experimentation

