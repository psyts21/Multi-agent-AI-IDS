import numpy as np
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix

class CoordinatorAgent:
    

    def __init__(self, agents: dict, threshold: float = 0.6, fallback_label: str = "normal"):
        self.agents = agents
        self.threshold = threshold  # This is now optional – not used in prediction logic below
        self.fallback_label = fallback_label.lower()

        # Include all agent names as attack types (no need to manually append fallback anymore)
        self.attack_types = list(agents.keys())

    def _score_agent(self, agent, sample_raw: np.ndarray) -> float:
        # wrap raw row → DataFrame (names help the scaler)
        sample_df = pd.DataFrame(sample_raw, columns=agent.feature_names)
        sample_scaled = pd.DataFrame(
            agent.scaler.transform(sample_df),
            columns=agent.feature_names
        )

        # pass the DataFrame (with names) straight to the model
        try:
            return agent.model.predict_proba(sample_scaled)[0, 1]
        except AttributeError:
            return float(agent.model.predict(sample_scaled)[0])
    
    def predict(self, X: np.ndarray) -> list[str]:
        X = np.asarray(X)
        preds = []

        for sample_raw in X:
            sample_raw = sample_raw.reshape(1, -1)

            confidences = {
                name: self._score_agent(agent, sample_raw)
                for name, agent in self.agents.items()
            }

            best_attack = max(confidences.items(), key=lambda kv: kv[1])[0]
            preds.append(best_attack)

        return preds

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> None:
        y_pred = self.predict(X)
        print("\nCoordinator Agent evaluation")
        print(confusion_matrix(y_true, y_pred, labels=self.attack_types))
        print(classification_report(y_true, y_pred, labels=self.attack_types))