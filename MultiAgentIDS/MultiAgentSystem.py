import pandas as pd
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from DetectionAgents import DetectionAgents

class MultiAgentSystem:
    def __init__(self, train_csv, test_csv):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.train_df = None
        self.test_df = None
        self.agents = {}

    def load_data_once(self):
        print("Loading and preprocessing shared data")
        self.train_df = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.test_csv)

        cols_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        self.train_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
        self.test_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    def setup_agents(self):
        print("running agents")
        self.agents = {
            "DoS": DetectionAgents(
                train_df=self.train_df,
                test_df=self.test_df,
                target_attack="DoS",
                model=XGBClassifier(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=2,
                    eval_metric="logloss",
                    verbosity=0
                )
            ),
            "Probe": DetectionAgents(
                train_df=self.train_df,
                test_df=self.test_df,
                target_attack="Probe",
                model=MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
            ),
            "R2L": DetectionAgents(
                train_df=self.train_df,
                test_df=self.test_df,
                target_attack="R2L",
                model=MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
            ),
            "U2R": DetectionAgents(
                train_df=self.train_df,
                test_df=self.test_df,
                target_attack="U2R",
                model=MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
            )
        }

    def run_system(self):
        results = []

        for name, agent in self.agents.items():
            print(f" {name} Agent ")

            agent.load_and_prepare()

            if not agent.load_model():
                agent.train()
                agent.save_model()

            result = agent.evaluate()
            results.append(result)

