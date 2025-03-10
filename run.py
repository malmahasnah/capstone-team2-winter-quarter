import sys
from causal_analysis import load_and_preprocess_data, run_pc_algorithm, visualize_causal_graph, compute_counterfactuals, run_fast_kci, run_kci_algorithm

def main(targets):
    # Predefined file path
    data_file_path = "data/Student Depression Dataset.csv"

    # Ensure at least two arguments are provided: mode and a parameter
    if len(targets) != 2:
        print("Error: You must provide a mode and a corresponding argument.")
        print("Usage:")
        print("  python run.py graph <method>            # Run PC algorithm (method: fisherz or chisq)")
        print("  python run.py counterfactual <variable> # Run counterfactual analysis on a variable")
        sys.exit(1)

    mode, param = targets

    if mode == "graph":
        # Validate causal discovery method
        if param not in ["fisherz", "chisq", "fastkci", "kci"]:
            print("Error: Invalid method. Choose either 'fisherz', 'chisq', 'fastkci', or 'kci'.")
            sys.exit(1)

        # Load and preprocess data
        selected_columns = ['Academic Pressure', 'Have you ever had suicidal thoughts ?', 'Financial Stress', 
                            'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad', 'City_Hyderabad', 'City_Meerut', 
                            'City_Patna', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 
                            'Dietary Habits_Unhealthy', 'Depression']
        data_encoded, data_array = load_and_preprocess_data(data_file_path, selected_columns)

        if param in ["fastkci", "kci"]:
            if param == "fastkci":
                run_fast_kci(data_encoded)
            elif param == "kci":
                run_kci_algorithm(data_encoded)
                # Create node labels
                node_labels = {i: col for i, col in enumerate(data_encoded.columns)}

                # Visualize Graph
                visualize_causal_graph(pc_graph, node_labels)
            
        if param in ["fisherz", "chisq"]:
            # Run PC Algorithm
            alpha = 0.05
            pc_graph = run_pc_algorithm(data_array, alpha, method=param)

            # Create node labels
            node_labels = {i: col for i, col in enumerate(data_encoded.columns)}

            # Visualize Graph
            visualize_causal_graph(pc_graph, node_labels)
            

    elif mode == "counterfactual":
        # Run counterfactual analysis on the specified variable
        compute_counterfactuals(data_file_path, param)

    else:
        print("Error: Invalid mode. Choose 'graph' or 'counterfactual'.")
        sys.exit(1)

if __name__ == '__main__':
    # run via:
    #   python run.py graph fisherz
    #   python run.py counterfactual "Academic Pressure"    
    targets = sys.argv[1:]
    main(targets)
