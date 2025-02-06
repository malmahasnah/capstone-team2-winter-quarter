import sys
from causal_analysis import load_and_preprocess_data, run_pc_algorithm, visualize_causal_graph

def main(targets):
    # Predefined file path
    data_file_path = "Student Depression Dataset.csv"

    # Validate that only one argument (model method) is provided
    if len(targets) != 1:
        print("Error: You must provide the model method.")
        print("Usage: python run.py <method>")
        sys.exit(1)

    method = targets[0]

    # Validate the method argument
    if method not in ["fisherz", "chisq"]:
        print("Error: Invalid method. Choose either 'fisherz' or 'chisq'.")
        sys.exit(1)

    # Columns to use for causal analysis (can be customized as needed)
    selected_columns = ['Academic Pressure', 'Have you ever had suicidal thoughts ?', 'Financial Stress', 
                        'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad', 'City_Hyderabad', 'City_Meerut', 
                        'City_Patna', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 
                        'Dietary Habits_Unhealthy', 'Depression']

    # Load and preprocess data
    data_encoded, data_array = load_and_preprocess_data(data_file_path, selected_columns)

    # Run PC Algorithm with selected method
    alpha = 0.05
    pc_graph = run_pc_algorithm(data_array, alpha, method=method)

    # Create node labels
    node_labels = {i: col for i, col in enumerate(data_encoded.columns)}

    # Visualize Graph
    visualize_causal_graph(pc_graph, node_labels)

if __name__ == '__main__':
    # run via:
    # python run.py <method>
    targets = sys.argv[1:]
    main(targets)
