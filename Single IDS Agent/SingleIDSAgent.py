import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

class SingleAgentIDS:
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

        # model and scaler
        self.random_forest = None 
        self.scaler = MinMaxScaler()

    def loading_datasets(self):
        self.dataset1 = pd.read_csv(self.training_data)
        self.dataset2 = pd.read_csv(self.test_data)

        columns_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        self.dataset1.drop(columns=columns_to_drop, inplace=True)
        self.dataset2.drop(columns=columns_to_drop, inplace=True)

        # features and labels
        self.X_train = self.dataset1.drop('attack_category', axis=1)
        self.y_train = self.dataset1['attack_category'].astype(str)
        self.X_test = self.dataset2.drop('attack_category', axis=1)
        self.y_test = self.dataset2['attack_category'].astype(str)

    
    def scaling_values(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def balancing_values(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)


    def tune_hyperparameters(self):
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_features': ['sqrt', 'log2', 0.3],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 5, 10]
        }

        randomized_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
            param_distributions=param_dist,
            n_iter=10, 
            cv=3,       
            scoring='f1_weighted',
            n_jobs=1,  
            verbose=1,
            random_state=42
        )

        X_sample = self.X_train[:100]
        y_sample = self.y_train[:100]

        randomized_search.fit(X_sample, y_sample)
        print("\n Best parameters:", randomized_search.best_params_)
        self.random_forest = randomized_search.best_estimator_

    def training_model(self):
        if not self.random_forest:
            self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.random_forest.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        print("\n TEST set:")
        y_pred_test = self.random_forest.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred_test))
        print(classification_report(self.y_test, y_pred_test))

        print("\n TRAINING set:")
        y_pred_train = self.random_forest.predict(self.X_train)
        print(confusion_matrix(self.y_train, y_pred_train))
        print(classification_report(self.y_train, y_pred_train))


# pipeline
if __name__ == "__main__":
    ids = SingleAgentIDS("processed_dataset.csv", "processed_dataset_test.csv")
    ids.loading_datasets()
    ids.scaling_values()
    ids.balancing_values()
    ids.tune_hyperparameters()
    ids.training_model()
    ids.evaluate_model()