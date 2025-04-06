import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import random

class SingleAgentIDS:
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

        self.random_forest = None
        self.xgb_model = None
        self.voting_clf = None
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def loading_datasets(self):
        self.dataset1 = pd.read_csv(self.training_data)
        self.dataset2 = pd.read_csv(self.test_data)

        columns_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        self.dataset1.drop(columns=columns_to_drop, inplace=True)
        self.dataset2.drop(columns=columns_to_drop, inplace=True)

        self.X_train = self.dataset1.drop('attack_category', axis=1)
        self.y_train_raw = self.dataset1['attack_category'].astype(str)
        self.X_test = self.dataset2.drop('attack_category', axis=1)
        self.y_test_raw = self.dataset2['attack_category'].astype(str)

        self.y_train = self.label_encoder.fit_transform(self.y_train_raw)
        self.y_test = self.label_encoder.transform(self.y_test_raw)

    def scaling_values(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def balancing_values(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def get_stratified_sample(self, size=1000):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
        for train_idx, _ in splitter.split(self.X_train, self.y_train):
            return self.X_train[train_idx], self.y_train[train_idx]

    # def tune_hyperparameters(self):
    #     param_dist = {
    #         'n_estimators': [100, 200, 300],
    #         'max_features': ['sqrt', 'log2', 0.3],
    #         'max_depth': [5, 10, 20, 30],
    #         'min_samples_leaf': [1, 5, 10]
    #     }

        # X_sample, y_sample = self.get_stratified_sample(size=1000)

        # randomized_search = RandomizedSearchCV(
        #     estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        #     param_distributions=param_dist,
        #     n_iter=20,
        #     cv=4,
        #     scoring='f1_weighted',
        #     n_jobs=1,
        #     verbose=1,
        #     random_state=42
        # )

        # randomized_search.fit(X_sample, y_sample)
        # print("\ rf paramc:", randomized_search.best_params_)
        # self.random_forest = randomized_search.best_estimator_

    def training_model(self):
        if not self.random_forest:
            self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.random_forest.fit(self.X_train, self.y_train)

    # def tune_xgboost_hyperparameters(self):
    #     param_dist = {
    #         'n_estimators': [100, 200, 300],
    #         'max_depth': [3, 6, 10],
    #         'learning_rate': [0.01, 0.1, 0.2],
    #         'subsample': [0.7, 0.8, 1.0],
    #         'colsample_bytree': [0.7, 0.8, 1.0]
    #     }

    #     X_sample, y_sample = self.get_stratified_sample(size=1000)

    #     randomized_search = RandomizedSearchCV(
    #         estimator=XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0),
    #         param_distributions=param_dist,
    #         n_iter=10,
    #         cv=3,
    #         scoring='f1_weighted',
    #         n_jobs=1,
    #         verbose=1,
    #         random_state=42
    #     )

    #     randomized_search.fit(X_sample, y_sample)
    #     print("\n xgboost param:", randomized_search.best_params_)
    #     self.xgb_model = randomized_search.best_estimator_

    def training_xgboost(self):
        if not self.xgb_model:
            self.xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )
        self.xgb_model.fit(self.X_train, self.y_train)

    def train_final_model(self):
        self.voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.random_forest),
                ('xgb', self.xgb_model)
            ],
            voting='hard'
        )
        self.voting_clf.fit(self.X_train, self.y_train)

    def respond_to_attack(self, prediction_label):
        responses = {
            "DoS": "Throttle traffic from source IP.",
            "Probe": "Log and monitor source IP.",
            "R2L": "Block user and alert administrator.",
            "U2R": "Isolate machine and trigger lockdown.",
            "normal": "No action needed."
        }
        return responses.get(prediction_label, "No action configured.")

    def evaluate_final_model(self):
        print("\n voting model  test set:")
        y_pred_test = self.voting_clf.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred_test))
        print(classification_report(self.y_test, y_pred_test, target_names=self.label_encoder.classes_))

        print("\n voting model training set:")
        y_pred_train = self.voting_clf.predict(self.X_train)
        print(confusion_matrix(self.y_train, y_pred_train))
        print(classification_report(self.y_train, y_pred_train, target_names=self.label_encoder.classes_))


        random_indices = random.sample(range(len(self.y_test)), 20)

        for i in random_indices:
            actual = self.label_encoder.inverse_transform([self.y_test[i]])[0]
            predicted = self.label_encoder.inverse_transform([y_pred_test[i]])[0]
            reaction = self.respond_to_attack(predicted)
            print(f"[{i}] True: {actual.ljust(7)} | Predicted: {predicted.ljust(7)} | Reaction: {reaction}")

#  Pipeline
if __name__ == "__main__":
    ids = SingleAgentIDS("processed_dataset.csv", "processed_dataset_test.csv")
    ids.loading_datasets()
    ids.scaling_values()
    ids.balancing_values()

    # ids.tune_hyperparameters()
    ids.training_model()

    # ids.tune_xgboost_hyperparameters()
    ids.training_xgboost()

    ids.train_final_model()
    ids.evaluate_final_model()
