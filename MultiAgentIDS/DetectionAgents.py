import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

class DetectionAgents:
    def __init__(self, training_data, test_data, target_attack="DoS", model=None):
        self.training_data = training_data
        self.test_data = test_data
        self.target_attack = target_attack  # constructor , will pass the specific attack and the csv 

        # using rf as default in case no model is used 
        self.model = model or RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        # irrelevant columns
        columns_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        data = data.drop(columns=columns_to_drop, errors='ignore')

        # preparing the inputs and binary features 
        X = data.drop(columns=['attack_category'])
        y = data['attack_category'].apply(lambda x: 1 if x == self.target_attack else 0)
        return X, y

    def load_and_prepare(self):
        train_df = pd.read_csv(self.training_data)
        test_df = pd.read_csv(self.test_data)

        self.X_train, self.y_train = self.preprocess_data(train_df)
        self.X_test, self.y_test = self.preprocess_data(test_df)

        # Normalising features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # SMOTE to balance the training set
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print(f"\n=== Evaluation Report for {self.target_attack} Agent ===")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred, target_names=[f"Not-{self.target_attack}", self.target_attack]))

    def predict_batch(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    


if __name__ == "__main__":
        print("test")

        dos_agent = DetectionAgents(
            training_data="processed_dataset.csv",
            test_data="processed_dataset_test.csv",
            target_attack="DoS"
        )

        dos_agent.load_and_prepare()
        dos_agent.train()
        dos_agent.evaluate()
