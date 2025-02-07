#hbrownell-261651
#This file is to be run on dsmlp servers in the background

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import us  # Import the us library

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

depression = pd.read_csv("Student Depression Dataset.csv")
depression = depression.dropna()
depression = depression.replace({'Yes': 1, 'No': 0})

data_encoded = pd.get_dummies(depression, drop_first=True)
data_encoded = data_encoded[[
    'Academic Pressure',
    'Have you ever had suicidal thoughts ?',
    'Financial Stress',
    'City_Bhavna',
    'City_Kibara',
    'City_ME',
    'City_Mira',
    'City_Nalyan',
    'City_Nandini',
    'City_Saanvi',
    'City_Vaanya',
    'Profession_Civil Engineer',
    'Profession_Digital Marketer',
    'Profession_Doctor',
    'Profession_Manager',
    'Profession_Student',
    'Profession_Teacher',
    'Dietary Habits_Moderate',
    'Dietary Habits_Others',
    'Dietary Habits_Unhealthy',
    'Degree_Others',
    'Depression'
]]

data_encoded = data_encoded.astype(int)
data_frame = data_encoded.sample(frac=.018)

# Step 1: Convert DataFrame to NumPy array>
data_array = data_frame.to_numpy()

# Step 2: Apply the PC algorithm to discover the causal graph
alpha = 0.05  # Significance level
pc_graph = pc(data_array, alpha, indep_test='kci')


# Step 3: Create labels for nodes based on DataFrame columns
node_labels = {i: col for i, col in enumerate(data_encoded.columns)}

# Step 4: Extract edges from the pc_graph and create a NetworkX directed graph
G = nx.DiGraph()
G.add_nodes_from(node_labels.keys())


with open("output.txt", "w") as f:
    for i in pc_graph.G.graph:
        f.write(str(i))