import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

from causallearn.search.ConstraintBased.PC import pc

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

def main():
    depression = pd.read_csv("Student Depression Dataset.csv")
    depression = depression.replace({'Yes': 1, 'No': 0})
    depression = depression.dropna()
    data_encoded = pd.get_dummies(depression, drop_first=True, dtype= int)
    data_encoded = data_encoded[['Academic Pressure', 'Have you ever had suicidal thoughts ?',
 'Financial Stress', 'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad',
 'City_Hyderabad', 'City_Meerut', 'City_Patna', 'Dietary Habits_Moderate',
 'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Depression']]

    data_frame = data_encoded
    # Step 1: Convert DataFrame to NumPy array>
    data_array = data_frame.to_numpy()
    
    # Step 2: Apply the PC algorithm to discover the causal graph
    alpha = 0.05  # Significance level
    pc_graph = pc(data_array, alpha)
    
    # Step 3: Create labels for nodes based on DataFrame columns
    node_labels = {i: col for i, col in enumerate(data_encoded.columns)}
    
    # Step 4: Extract edges from the pc_graph and create a NetworkX directed graph
    G = nx.DiGraph()
    G.add_nodes_from(node_labels.keys())
    
    # Add edges based on the adjacency matrix
    for i in range(len(pc_graph.G.graph)):
        for j in range(len(pc_graph.G.graph)):
            if pc_graph.G.graph[i, j] != 0:  # Check for an edge
                if pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == -1:
                    # Case: i -> j
                    G.add_edge(i, j, edge_type='directed')
                elif pc_graph.G.graph[j, i] == -1 and pc_graph.G.graph[i, j] == -1:
                    # Case: i -- j (undirected)
                    G.add_edge(i, j, edge_type='undirected')
                elif pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == 1:
                    # Case: i <-> j (bidirectional)
                    G.add_edge(i, j, edge_type='bidirectional')
    # Step 5: Plot the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # Increase k for more space between nodes
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, node_color="skyblue", font_size=7, font_weight="bold", arrowstyle="->", arrowsize=20)
    plt.title("Inferred Causal Graph")
    plt.show()
    
if __name__ == "__main__":
    main()