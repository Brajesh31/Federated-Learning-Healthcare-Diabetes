# server.py
import flwr as fl
import matplotlib.pyplot as plt

# Store accuracy after each round
round_accuracies = []

# Custom metric aggregation function
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    round_accuracy = sum(accuracies) / total_examples

    # Store accuracy for plotting later
    round_accuracies.append(round_accuracy)
    return {"accuracy": round_accuracy}

# Define FL strategy with metric aggregation
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average
)

# Train the FL server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

# After training finishes, plot the accuracy over rounds
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(round_accuracies) + 1), round_accuracies, marker='o', color='blue')
plt.title("Federated Learning Accuracy over Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("federated_accuracy_plot.png")  # Save as image (optional)
plt.show()
