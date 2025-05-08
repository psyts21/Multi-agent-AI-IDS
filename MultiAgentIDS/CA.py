import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class CoordinatorAgent:
    def __init__(self, agents: dict, threshold: float = 0.6, fallback_label: str = "normal"):
        self.agents = agents
        self.threshold = threshold
        self.fallback_label = fallback_label.lower()
        self.attack_types = list(agents.keys())

        # You can tweak these weights based on experimental results
        self.weights = {
            "DoS": 0.003,
            "normal": 0.002,
            "Probe": 0.012,
            "R2L": 0.002,
            "U2R": 0.048,
        }

    def _score_agent(self, agent, sample_raw: np.ndarray) -> float:
        sample_df = pd.DataFrame(sample_raw, columns=agent.feature_names)
        sample_scaled = pd.DataFrame(agent.scaler.transform(sample_df), columns=agent.feature_names)
        try:
            return agent.model.predict_proba(sample_scaled)[0, 1]
        except AttributeError:
            return float(agent.model.predict(sample_scaled)[0])

    def _get_confidence_vector(self, sample_raw: np.ndarray) -> np.ndarray:
        return np.array([
            self._score_agent(agent, sample_raw)
            for _, agent in self.agents.items()
        ])

    def predict(self, X: np.ndarray) -> list[str]:
        X = np.asarray(X)
        preds = []

        for sample_raw in X:
            sample_raw = sample_raw.reshape(1, -1)
            confidence_vector = self._get_confidence_vector(sample_raw)


            weighted_scores = {
                attack: score * self.weights.get(attack, 1.0)
                for attack, score in zip(self.attack_types, confidence_vector)
            }

            # Choose the label with the highest weighted score
            pred_label = max(weighted_scores, key=weighted_scores.get)
            preds.append(pred_label)

        return preds

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = self.predict(X)
        print("\nCoordinator Agent evaluation (Weighted Voting)")
        print(confusion_matrix(y_true, y_pred, labels=self.attack_types))
        print(classification_report(y_true, y_pred, labels=self.attack_types))
