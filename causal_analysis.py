import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc

def load_and_preprocess_data(file_path, selected_columns=None):
    """
    Loads and preprocesses the dataset.

    :param file_path: Path to the dataset CSV file.
    :param selected_columns: List of columns to include (optional).
    :return: Processed DataFrame and NumPy array for causal analysis.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Convert categorical yes/no to binary
    df = df.replace({'Yes': 1, 'No': 0})
    
    # Drop missing values
    df = df.dropna()

    # Encode categorical variables
    data_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    # Select only specified columns if provided
    if selected_columns:
        data_encoded = data_encoded[selected_columns]

    return data_encoded, data_encoded.to_numpy()

def run_pc_algorithm(data_array, alpha=0.05):
    """
    Runs the PC algorithm on the given dataset.

    :param data_array: NumPy array containing the processed data.
    :param alpha: Significance level for conditional independence tests.
    :return: Learned causal graph.
    """
    return pc(data_array, alpha)

def visualize_causal_graph(pc_graph, node_labels):
    """
    Visualizes the causal graph using NetworkX.

    :param pc_graph: Output from the PC algorithm.
    :param node_labels: Dictionary mapping node indices to column names.
    """
    G = nx.DiGraph()
    G.add_nodes_from(node_labels.keys())

    # Add edges based on PC adjacency matrix
    for i in range(len(pc_graph.G.graph)):
        for j in range(len(pc_graph.G.graph)):
            if pc_graph.G.graph[i, j] != 0:  # Edge exists
                if pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == -1:
                    G.add_edge(i, j, edge_type='directed')  # i -> j
                elif pc_graph.G.graph[j, i] == -1 and pc_graph.G.graph[i, j] == -1:
                    G.add_edge(i, j, edge_type='undirected')  # i -- j
                elif pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == 1:
                    G.add_edge(i, j, edge_type='bidirectional')  # i <-> j

    # Plot the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, node_color="skyblue",
            font_size=8, font_weight="bold", arrowstyle="->", arrowsize=15)
    plt.title("Inferred Causal Graph")
    plt.show()
