import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from collections import Counter
import os
import joblib

class DetectionAgents:
    def __init__(self, train_df, test_df, target_attack="DoS", model=None):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.target_attack = target_attack

        self.model = model or XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=2,
            eval_metric="logloss",
            verbosity=0
        )
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        columns_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        data = data.drop(columns=columns_to_drop, errors='ignore')
        X = data.drop(columns=['attack_category'])
        y = data['attack_category'].apply(lambda x: 1 if x == self.target_attack else 0)
        return X, y
    

    

    def load_and_prepare(self):
        self.X_train, self.y_train = self.preprocess_data(self.train_df)
        self.X_test, self.y_test = self.preprocess_data(self.test_df)

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("before:", Counter(self.y_train))

        smote = SMOTETomek(smote=SMOTE(k_neighbors=3), random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)

        print("after:", Counter(y_resampled))

        self.X_train, self.y_train = X_resampled, y_resampled


    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print(f"\ Evaluation of the  {self.target_attack} agent ")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred, target_names=[f"Not-{self.target_attack}", self.target_attack]))

    
    def save_model(self, path="models"):
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/{self.target_attack}_model.pkl"
        joblib.dump(self.model, filename)
        print(f"saving model of {self.target_attack} in {filename}")

    def load_model(self, path="models"):
        filename = f"{path}/{self.target_attack}_model.pkl"
        if os.path.exists(filename):
            self.model = joblib.load(filename)
            print(f" fetching model of {self.target_attack}")
            return True
        print(f" No model found for {self.target_attack}")
        return False