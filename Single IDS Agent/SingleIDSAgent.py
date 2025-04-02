import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix


class SingleAgentIDS:
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

        # Initialize the model and scaler
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()

    def loading_datasets(self):
        # Load the processed datasets
        self.dataset1 = pd.read_csv(self.training_data)
        self.dataset2 = pd.read_csv(self.test_data)

        # Drop unnecessary columns
        columns_to_drop = ['protocol_type', 'service', 'flag', 'label', 'difficulty']
        self.dataset1.drop(columns=columns_to_drop, inplace=True)
        self.dataset2.drop(columns=columns_to_drop, inplace=True)

        # Define features (X) and labels (y)
        self.X_train = self.dataset1.drop('attack_category', axis=1)
        self.y_train = self.dataset1['attack_category'].astype(str)  # ensure consistent type
        self.X_test = self.dataset2.drop('attack_category', axis=1)
        self.y_test = self.dataset2['attack_category'].astype(str)

    def scaling_values(self):
    
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def training_model(self):
   
        self.random_forest.fit(self.X_train, self.y_train)

    def evaluate_model(self):
     
        print(" TEST set:")
        y_pred_test = self.random_forest.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred_test))
        print(classification_report(self.y_test, y_pred_test))


        print("\n🎓training set:")
        y_pred_train = self.random_forest.predict(self.X_train)
        print(confusion_matrix(self.y_train, y_pred_train))
        print(classification_report(self.y_train, y_pred_train))

#  pipeline
if __name__ == "__main__":
    ids = SingleAgentIDS("processed_dataset.csv", "processed_dataset_test.csv")
    ids.loading_datasets()
    ids.scaling_values()
    ids.training_model()
    ids.evaluate_model()
