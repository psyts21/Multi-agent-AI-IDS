import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class CoordinatorAgent:

    def __init__(self, agents, threshold=0.6):
       
        self.agents = agents  # dictionary of agents
        self.attack_types = list(agents.keys())
        self.threshold = threshold  # confidence threshold 

    def predict(self, X):
    
        final_predictions = []
# looping over the samples
        for i in range(len(X)):
            sample = X[i].reshape(1, -1)
            confidences = {}
   # communicating with each agents
            for attack, agent in self.agents.items():
                try:
            
                    prob = agent.model.predict_proba(sample)[0][1]
                except AttributeError:
                    prob = agent.model.predict(sample)[0]  # fallback 
                confidences[attack] = prob

            # agent with the highest confidence score
            best_attack = max(confidences, key=confidences.get)
            best_score = confidences[best_attack]

            if best_score >= self.threshold:
                final_predictions.append(best_attack)
            else:
                final_predictions.append("Normal")

        return final_predictions

    def evaluate(self, X, y_true):
 
        y_pred = self.predict(X)
        print("\Coordinator Agent evaluation")
        print(confusion_matrix(y_true, y_pred, labels=self.attack_types + ["Normal"]))
        print(classification_report(y_true, y_pred, labels=self.attack_types + ["Normal"]))
