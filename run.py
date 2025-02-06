from causal_analysis import load_and_preprocess_data, run_pc_algorithm, visualize_causal_graph

def main():
    # File path (modify as needed)
    file_path = "Student Depression Dataset.csv"

    # Columns to use for causal analysis
    selected_columns = ['Academic Pressure', 'Have you ever had suicidal thoughts ?', 'Financial Stress', 
                        'City_Ahmedabad', 'City_Bhopal', 'City_Faridabad', 'City_Hyderabad', 'City_Meerut', 
                        'City_Patna', 'Dietary Habits_Moderate', 'Dietary Habits_Others', 
                        'Dietary Habits_Unhealthy', 'Depression']

    # Load and preprocess data
    data_encoded, data_array = load_and_preprocess_data(file_path, selected_columns)

    # Run PC Algorithm
    alpha = 0.05
    pc_graph = run_pc_algorithm(data_array, alpha)

    # Create node labels
    node_labels = {i: col for i, col in enumerate(data_encoded.columns)}

    # Visualize Graph
    visualize_causal_graph(pc_graph, node_labels)

# Ensures script runs only when executed directly, not when imported
if __name__ == "__main__":
    main()
