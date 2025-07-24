from network import GermanWordsNetwork
import utils
from pathlib import Path

def main():
    print("=== German Words Network Visualization ===\n")
    network = GermanWordsNetwork()
    
    # Try to load from SQLite, then CSV, then fallback to sample data
    data_loaded = False
    db_path = "data/corpus.sqlite3"
    csv_path = "data/articles.csv"
    
    if Path(db_path).exists():
        print(f"Found database at {db_path}, loading data...")
        network.load_from_sqlite(db_path, limit=10)
        data_loaded = True
    elif Path(csv_path).exists():
        print(f"Found CSV file at {csv_path}, loading data...")
        network.load_from_csv(csv_path, limit=1000)
        data_loaded = True
    if not data_loaded:
        print("No data files found, using sample German text...")
        sample_sentences = utils.generate_sample_data()
        for sentence in sample_sentences:
            for w1, w2 in utils.extract_word_pairs(sentence):
                network.add_word_pair(w1, w2)
                
    if network.has_data():  # Ensure the network has data before building the graph
        network.build_graph()
    network.visualize("german_words_network.html")
    
    # Only extract paths if the graph has enough nodes/edges
    if hasattr(network, 'graph') and len(network.graph.nodes) > 1 and len(network.graph.edges) > 0:
        paths = network.extract_unique_paths(max_depth=10)
        if paths:
            network.plot_paths_as_binary_tree(paths, max_paths=3)
        else:
            print("No unique paths found to plot as a binary tree.")
    
    print("\n=== Statistics ===")
    stats = network.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("\n=== Statistics ===")
    stats = network.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
