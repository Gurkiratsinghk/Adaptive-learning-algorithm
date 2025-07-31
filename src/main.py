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
        network.load_from_sqlite(db_path, limit=1000)  # Increased limit for better anchor detection
        data_loaded = True
    elif Path(csv_path).exists():
        print(f"Found CSV file at {csv_path}, loading data...")
        network.load_from_csv(csv_path, limit=1000)
        data_loaded = True
    
    if not data_loaded:
        print("No data files found, using sample German text...")
        sample_sentences = utils.generate_sample_data()
        for sentence in sample_sentences:
            pairs = utils.extract_word_pairs(sentence)
            sentence_starts = utils.extract_sentence_starts(sentence)
            
            for w1, w2 in pairs:
                network.add_word_pair(w1, w2)
            
            for start_word in sentence_starts:
                network.sentence_starts[start_word] += 1
    
    if network.has_data():  # Ensure the network has data before building the graph
        print("\n=== Building Anchor-Based Graph ===")
        # Use the new anchor-based graph building method
        network.build_anchor_graph(max_nodes=100, min_edge_weight=2)
        
        # Visualize the anchor network
        network.visualize("german_anchor_network.html")
        print("Network visualization saved to german_anchor_network.html")
        
        print("\n=== Anchor Path Analysis ===")
        # Show some example anchor chains
        if hasattr(network, 'anchor_chains') and network.anchor_chains:
            print("Sample anchor chains:")
            for anchor, chains in list(network.anchor_chains.items())[:3]:
                print(f"\nAnchor '{anchor}':")
                for i, chain in enumerate(chains[:3]):  # Show top 3 chains per anchor
                    print(f"  Chain {i+1}: {' -> '.join(chain)}")
        
        # Show cross-chain connections
        if hasattr(network, 'cross_chain_connections') and network.cross_chain_connections:
            print(f"\nSample cross-chain connections:")
            for conn in network.cross_chain_connections[:5]:
                print(f"  {conn['from_anchor']}[{conn['from_word']}] -> {conn['to_anchor']}[{conn['to_word']}] (freq: {conn['frequency']})")
    
    else:
        print("No word pairs found in the data.")
    
    print("\n=== Network Statistics ===")
    stats = network.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()