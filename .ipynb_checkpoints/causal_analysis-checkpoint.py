import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from dowhy import gcm

def load_and_preprocess_data(file_path, selected_columns=None):
    df = pd.read_csv(file_path)
    df = df.replace({'Yes': 1, 'No': 0})
    df = df.dropna()
    data_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    if selected_columns:
        data_encoded = data_encoded[selected_columns]
    
    return data_encoded, data_encoded.to_numpy()

def run_pc_algorithm(data_array, alpha=0.05, method="fisherz"):
    if method not in ["fisherz", "chisq"]:
        raise ValueError("Invalid method. Choose 'fisherz' or 'chisq'.")
    
    return pc(data_array, alpha, indep_test=method)

def visualize_causal_graph(pc_graph, node_labels):
    G = nx.DiGraph()
    G.add_nodes_from(node_labels.keys())

    for i in range(len(pc_graph.G.graph)):
        for j in range(len(pc_graph.G.graph)):
            if pc_graph.G.graph[i, j] != 0:
                if pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == -1:
                    G.add_edge(i, j, edge_type='directed')
                elif pc_graph.G.graph[j, i] == -1 and pc_graph.G.graph[i, j] == -1:
                    G.add_edge(i, j, edge_type='undirected')
                elif pc_graph.G.graph[j, i] == 1 and pc_graph.G.graph[i, j] == 1:
                    G.add_edge(i, j, edge_type='bidirectional')
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000, node_color="skyblue",
            font_size=8, font_weight="bold", arrowstyle="->", arrowsize=15)
    plt.title("Inferred Causal Graph")
    plt.show()

def compute_counterfactuals1(file_path, intervention_var):
    df = pd.read_csv(file_path).dropna()
    df = df.replace({'Yes': 1, 'No': 0})
    data_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    selected_columns = ['Academic_Pressure', 'Suicidal_Thoughts',
                        'Financial_Stress', 'Depression']
    
    if intervention_var not in selected_columns:
        raise ValueError(f"Invalid intervention variable. Choose from: {selected_columns}")

    # Prompt user for observed values (excluding 'Depression')
    observed_data = {}
    print(f"\nPlease enter observed values for the following variables (excluding '{intervention_var}' and 'Depression'):")
    
    observed_data[intervention_var] = 1
    observed_data['Depression'] = 1
    
    for var in selected_columns:
            if var != intervention_var and var != 'Depression':  # Skip Depression
                while True:
                    try:
                        if var == 'Academic_Pressure' or var == 'Financial_Stress':
                            value = int(input(f"  {var} (1-5): "))
                            if value < 1 or value > 5:
                                print("Invalid input. Please enter a value between 1 and 5.")
                                continue
                        elif var == 'Suicidal_Thoughts':
                            value = int(input(f"  {var} (0-1): "))
                            if value < 0 or value > 1:
                                print("Invalid input. Please enter a value between 0 and 1.")
                                continue

                        observed_data[var] = value
                        break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
                        
    # Convert observed values into DataFrame (without Depression)
    observed_data_df = pd.DataFrame([observed_data])

    # Construct causal model
    causal_model = gcm.InvertibleStructuralCausalModel(nx.DiGraph([
                                        ('Financial_Stress', 'Depression'),
                                        ('Suicidal_Thoughts', 'Depression'),
                                        ('Academic_Pressure', 'Depression')
                                        ])
                                        )
    causal_model.set_causal_mechanism('Financial_Stress', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Suicidal_Thoughts', 
                                      gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Academic_Pressure', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Depression', 
                                      gcm.AdditiveNoiseModel(
                                          gcm.ml.create_linear_regressor()))
        
    # Train model
    training_data = pd.DataFrame(data=dict(
    **{'Financial_Stress': data_encoded['Financial Stress']},
    **{'Suicidal_Thoughts': data_encoded['Have you ever had suicidal thoughts ?']},
    **{'Academic_Pressure': data_encoded['Academic Pressure']},
    **{'Depression': data_encoded['Depression']}
    ))
    
    gcm.fit(causal_model, training_data)

    
    # Dynamically set the range of values for the intervention variable
    if intervention_var == 'Suicidal_Thoughts':
        intervention_range = [0, 1]  # Only 0 or 1 for Suicidal Thoughts
    elif intervention_var == 'Academic_Pressure' or intervention_var == 'Financial_Stress':
        intervention_range = range(1, 6)  # 1-5 for Academic Pressure and Financial Stress
    else:
        raise ValueError(f"Unsupported intervention variable: {intervention_var}")
    
    # Generate counterfactuals for different intervention values
    print(f"\nIntervening on '{intervention_var}' with different values...")
    
    counterfactual_results = {}
    for value in intervention_range:  
        result = gcm.counterfactual_samples(
            causal_model,
            {intervention_var: lambda x: value},  
            observed_data=observed_data_df
        )

        counterfactual_results[f"{intervention_var}={value}"] = result['Depression'].values[0]
    # Display results
    counterfactual_df = pd.DataFrame(counterfactual_results, index=['Depression'])
    print("\nCounterfactual Results:")
    print(counterfactual_df)

    return counterfactual_df

