#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import us  # Import the us library

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import     orient_by_background_knowledge

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

from causallearn.utils.KCI.KCI import KCI_CInd

from causallearn.search.ConstraintBased.FCI import fci


# In[ ]:


depression = pd.read_csv("data/Student Depression Dataset.csv")
depression = depression.dropna()
depression = depression.replace({'Yes': 1, 'No': 0})

data_encoded = pd.get_dummies(depression, drop_first=True)
data_encoded = data_encoded[['Academic Pressure', 'Have you ever had suicidal thoughts ?',
 'Financial Stress', 'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad',
 'City_Hyderabad', 'City_Meerut', 'City_Patna', 'Dietary Habits_Moderate',
 'Dietary Habits_Others', 'Dietary Habits_Unhealthy', 'Depression']]
data_encoded = data_encoded.astype(int)


# In[6]:


data_encoded = data_encoded[:2000]
data_encoded


# In[7]:


data_matrix = data_encoded.values
data_matrix



from causallearn.utils.GraphUtils import GraphUtils


# In[9]:

g, edges = fci(data_matrix, independence_test_method='kci')


pdy = GraphUtils.to_pydot(g)


# In[15]:


# Convert the causal graph to a networkx graph
G = nx.DiGraph()  # Use DiGraph for directed edges

variable_names = data_encoded.columns.tolist()

# Get the adjacency matrix from the PC output
adj_matrix = g.graph

num_nodes = adj_matrix.shape[0]
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj_matrix[i, j] != 0:  # If there's an edge
            G.add_edge(variable_names[i], variable_names[j])  # Use column names as labels

# Add edges based on adjacency matrix
# num_nodes = adj_matrix.shape[0]
# for i in range(num_nodes):
#    for j in range(num_nodes):
#        if adj_matrix[i, j] != 0:  # If there's an edge
#            G.add_edge(i, j)  # Directed edge from i to j

# Plot the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", 
        node_size=2000, font_size=7, arrows=True)

pos = nx.circular_layout(G)

edge_labels = {(variable_names[i], variable_names[j]): "" for i in range(num_nodes) for j in range(num_nodes) if adj_matrix[i, j] != 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Causal Graph using Fast KCI")
plt.show()






