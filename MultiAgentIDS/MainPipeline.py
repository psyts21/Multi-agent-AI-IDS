from MultiAgentSystem import MultiAgentSystem
from CA import CoordinatorAgent
import pandas as pd

from sklearn.metrics import classification_report


if __name__ == "__main__":
    print("Multi-Agent IDS System")

    system = MultiAgentSystem(
        train_csv="processed_dataset.csv",
        test_csv="processed_dataset_test.csv"
    )

    system.load_data_once()
    system.setup_agents()
    system.run_system()



    X_test = system.test_df.drop(columns=["attack_category"])
    y_true = system.test_df["attack_category"].values
    scaler = system.agents["DoS"].scaler
    X_scaled = scaler.transform(X_test)



    coordinator = CoordinatorAgent(system.agents, threshold=0.6)
    coordinator.evaluate(X_scaled, y_true)

    report = classification_report(
        y_true, 
        coordinator.predict(X_scaled), 
        labels=coordinator.attack_types + ["Normal"], 
        output_dict=True
    )
    df = pd.DataFrame(report).transpose()
    df.to_csv("evaluation_multiagent.csv")
    print(" saved csv")