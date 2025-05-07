import pandas as pd
import json
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from DetectionAgents import DetectionAgents

if __name__ == "__main__":
    print(" Running hyperparameter tuning for all agents")

    # Load the preprocessed data
    train_df = pd.read_csv("processed_dataset.csv")
    test_df = pd.read_csv("processed_dataset_test.csv")

    #  grids for xgb 
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 2]
    }


    #  grids for NN
    mlp_param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'solver': ['adam'],
        'max_iter': [300]
    }

    # configs
    agent_configs = [
        {
            "name": "DoS",
            "model": XGBClassifier(eval_metric="logloss", verbosity=0),
            "param_grid": xgb_param_grid
        },
        {
            "name": "Probe",
            "model": MLPClassifier(random_state=42),
            "param_grid": mlp_param_grid
        },
        {
            "name": "R2L",
            "model": MLPClassifier(random_state=42),
            "param_grid": mlp_param_grid
        },
        {
            "name": "U2R",
            "model": MLPClassifier(random_state=42),
            "param_grid": mlp_param_grid
        }
    ]

    best_params = {}

    for config in agent_configs:
        print(f"\n Tuning {config['name']} the agent ...")
        agent = DetectionAgents(
            train_df=train_df,
            test_df=test_df,
            target_attack=config["name"],
            model=config["model"],
            param_grid=config["param_grid"],
            tune_hyperparameters=True
        )

        agent.load_and_prepare()

        search = RandomizedSearchCV(
            estimator=agent.model,
            param_distributions=config["param_grid"],
            n_iter=10,
            scoring='accuracy',
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        search.fit(agent.X_train, agent.y_train)

        best_params[config["name"]] = search.best_params_

        # Print best hyperparameters
        print(f"\n  Best hyperparameters for {config['name']}:")
        for param, value in search.best_params_.items():
            print(f"  - {param}: {value}")

    # saving 
    with open("best_hyperparams.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("\n  All best hyperparameters saved to best_hyperparams.json ")
