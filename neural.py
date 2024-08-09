import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def runscript(csv_file_path):
    # Update this path to the location of your file
    df = pd.read_csv(csv_file_path)

    # Replace specific values with 'No'
    df = df.replace({'No phone service': 'No', 'No internet service': 'No'})

    # Convert string values to integers or floats, or factorize if not possible
    def str_to_int(data):
        for col in data.columns:
            for i, item in enumerate(data[col]):
                if isinstance(item, str):
                    try:
                        data.at[i, col] = float(item)
                    except ValueError:
                        if data[col].dtype == 'object':
                            labels, _ = pd.factorize(data[col])
                            data[col] = labels
        return data

    df = str_to_int(df)

    # Remove rows with empty strings
    def remove_rows_with_empty_strings(df):
        mask = df.apply(lambda x: x == ' ')
        rows_with_empty_strings = mask.any(axis=1)
        df_cleaned = df[~rows_with_empty_strings]
        return df_cleaned

    df = remove_rows_with_empty_strings(df)
    dfn = df.sample(n=1300, random_state=42)

    # Split the data into features and target variable
    X = dfn.drop(["customerID", "Churn"], axis=1)
    y = dfn["Churn"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(3, 5, 15), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    # Predict and calculate accuracy
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Function to plot the MLPClassifier structure
    def plot_mlp_structure(mlp, input_size):
        layers = [input_size] + list(mlp.hidden_layer_sizes) + [1]  # Add input and output layers

        G = nx.DiGraph()
        node_count = 0
        layer_nodes = []

        # Create nodes for each layer
        for i, layer_size in enumerate(layers):
            layer_nodes.append([])
            for _ in range(layer_size):
                G.add_node(node_count, layer=i)
                layer_nodes[-1].append(node_count)
                node_count += 1

        # Create edges between nodes of subsequent layers
        for i in range(len(layer_nodes) - 1):
            for node in layer_nodes[i]:
                for next_node in layer_nodes[i + 1]:
                    G.add_edge(node, next_node)

        pos = {}
        for i, layer in enumerate(layer_nodes):
            for j, node in enumerate(layer):
                pos[node] = (i, j - len(layer) / 2)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_size=500, node_color="skyblue", edge_color="gray")
        labels = {i: f"Input {i+1}" for i in range(input_size)}
        hidden_layers = sum([[f"H{j+1}" for j in range(size)] for size in mlp.hidden_layer_sizes], [])
        labels.update({i + input_size: hidden_layers[i] for i in range(len(hidden_layers))})
        labels.update({node_count - 1: "Output"})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        plt.title("MLPClassifier Structure")
        plt.savefig("mlp_structure_plot.png")  # Save the plot as a file
        plt.close()

    # Plot the MLPClassifier structure
    plot_mlp_structure(mlp, X.shape[1])
    return accuracy, "mlp_structure_plot.png"

