from MultiAgentSystem import MultiAgentSystem

if __name__ == "__main__":
    print("=== Multi-Agent IDS System ===")

    system = MultiAgentSystem(
        train_csv="processed_dataset.csv",
        test_csv="processed_dataset_test.csv"
    )

    system.load_data_once()
    system.setup_agents()
    system.run_system()
