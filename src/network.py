from collections import defaultdict
from pyvis.network import Network
import pandas as pd
import sqlite3
from pathlib import Path
from utils import extract_word_pairs, extract_sentence_starts

class GermanWordsNetwork:
    """
    A class to manage a network of German words using Pyvis for visualization.
    Each node represents a unique German word, with directed edges indicating
    word sequence frequency.
    """
    def __init__(self):
        self.graph = None
        self.word_pairs = defaultdict(int)
        self.vocabulary = set()
        self.total_pairs = 0
        self.anchor_nodes = set()
        self.anchor_chains = {}
        self.cross_chain_connections = []
        self.sentence_starts = defaultdict(int)  # Track how often words start sentences

    def add_word_pair(self, word1: str, word2: str) -> None:
        self.word_pairs[(word1, word2)] += 1
        self.vocabulary.add(word1)
        self.vocabulary.add(word2)
        self.total_pairs += 1

    def identify_anchor_nodes(self, min_outgoing=2, min_ratio=1.5, min_frequency=3) -> None:
        """
        Identify anchor nodes that typically start sentences or have high outgoing ratios.
        
        Args:
            min_outgoing: Minimum number of outgoing edges
            min_ratio: Minimum ratio of outgoing to incoming edges
            min_frequency: Minimum total frequency for outgoing edges
        """
        print("Identifying anchor nodes...")
        
        # Calculate incoming and outgoing edges for each word
        outgoing = defaultdict(list)  # word -> [(target_word, frequency), ...]
        incoming = defaultdict(int)   # word -> total_incoming_frequency
        outgoing_freq = defaultdict(int)  # word -> total_outgoing_frequency
        
        for (word1, word2), freq in self.word_pairs.items():
            outgoing[word1].append((word2, freq))
            outgoing_freq[word1] += freq
            incoming[word2] += freq
        
        # Identify potential anchor nodes
        candidates = []
        
        for word in self.vocabulary:
            out_count = len(outgoing[word])
            out_freq = outgoing_freq[word]
            in_freq = incoming[word]
            sentence_start_freq = self.sentence_starts[word]
            
            # Skip if not enough outgoing connections
            if out_count < min_outgoing or out_freq < min_frequency:
                continue
            
            # Calculate ratio (add 1 to avoid division by zero)
            ratio = out_freq / (in_freq + 1)
            
            # Boost score for words that frequently start sentences
            sentence_boost = 1 + (sentence_start_freq * 0.5)
            adjusted_ratio = ratio * sentence_boost
            
            # Check if this could be an anchor node
            if adjusted_ratio >= min_ratio or in_freq == 0 or sentence_start_freq > 2:
                candidates.append((word, adjusted_ratio, out_freq, out_count, sentence_start_freq))
        
        # Sort by adjusted ratio and frequency
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Select top anchor nodes (limit to prevent too many)
        max_anchors = min(20, len(candidates))
        self.anchor_nodes = set(word for word, _, _, _, _ in candidates[:max_anchors])
        
        print(f"Identified {len(self.anchor_nodes)} anchor nodes:")
        for word in list(self.anchor_nodes)[:10]:  # Show first 10
            candidate_info = next((r, f, s) for w, r, _, f, s in candidates if w == word)
            ratio, freq, sent_freq = candidate_info
            print(f"  - '{word}' (ratio: {ratio:.2f}, out_freq: {freq}, sentence_starts: {sent_freq})")

    def build_anchor_chains(self, max_chain_length=5) -> None:
        """
        Build chains starting from each anchor node.
        """
        print("Building anchor chains...")
        
        # Build adjacency list for easier traversal
        adj = defaultdict(list)
        for (word1, word2), freq in self.word_pairs.items():
            adj[word1].append((word2, freq))
        
        self.anchor_chains = {}
        
        for anchor in self.anchor_nodes:
            chains = []
            
            def build_chain(current_word, current_chain, visited, remaining_depth):
                if remaining_depth <= 0 or current_word in visited:
                    if len(current_chain) > 1:  # Only keep chains with at least 2 words
                        chains.append(current_chain.copy())
                    return
                
                visited.add(current_word)
                
                # Get outgoing edges sorted by frequency
                neighbors = sorted(adj[current_word], key=lambda x: x[1], reverse=True)
                
                # Explore top neighbors
                for next_word, freq in neighbors[:3]:  # Limit to top 3 to avoid explosion
                    if next_word not in self.anchor_nodes:  # Don't connect to other anchors directly
                        current_chain.append(next_word)
                        build_chain(next_word, current_chain, visited.copy(), remaining_depth - 1)
                        current_chain.pop()
                
                # Also add single-step chains for high-frequency direct connections
                if len(current_chain) == 1:
                    for next_word, freq in neighbors[:5]:
                        if freq >= 2:  # Reduced threshold for more connections
                            chains.append([current_word, next_word])
            
            # Start building chains from the anchor
            if anchor in adj:  # Only build chains if anchor has outgoing edges
                build_chain(anchor, [anchor], set(), max_chain_length)
            
            # Ensure we have at least some chains for visualization
            if not chains and anchor in adj:
                # Add simple direct connections as fallback
                for next_word, freq in adj[anchor][:3]:
                    if freq >= 1:  # Very low threshold as fallback
                        chains.append([anchor, next_word])
            
            self.anchor_chains[anchor] = chains
        
        total_chains = sum(len(chains) for chains in self.anchor_chains.values())
        print(f"Built {total_chains} anchor chains:")
        for anchor, chains in list(self.anchor_chains.items())[:5]:  # Show first 5 anchors
            print(f"  - '{anchor}': {len(chains)} chains")

    def find_cross_chain_connections(self, min_frequency=2) -> None:
        """
        Find connections between different anchor chains based on frequency threshold.
        """
        print("Finding cross-chain connections...")
        
        self.cross_chain_connections = []
        
        # Create a mapping of words to their anchor chains
        word_to_chains = defaultdict(list)
        for anchor, chains in self.anchor_chains.items():
            for i, chain in enumerate(chains):
                for word in chain[1:]:  # Skip the anchor itself
                    word_to_chains[word].append((anchor, i, chain))
        
        # Find connections between chains
        for (word1, word2), freq in self.word_pairs.items():
            if freq >= min_frequency:
                # Check if word1 is in one anchor chain and word2 is in another
                if word1 in word_to_chains and word2 in word_to_chains:
                    for anchor1, chain1_idx, chain1 in word_to_chains[word1]:
                        for anchor2, chain2_idx, chain2 in word_to_chains[word2]:
                            if anchor1 != anchor2:  # Different anchor chains
                                connection = {
                                    'from_anchor': anchor1,
                                    'to_anchor': anchor2,
                                    'from_word': word1,
                                    'to_word': word2,
                                    'frequency': freq,
                                    'from_chain': chain1,
                                    'to_chain': chain2
                                }
                                self.cross_chain_connections.append(connection)
        
        print(f"Found {len(self.cross_chain_connections)} cross-chain connections")

    def build_anchor_graph(self, max_nodes=100, min_edge_weight=1, physics=False) -> None:
        """
        Build the anchor-based graph with anchor nodes and their chains.
        """
        print("Building anchor-based graph...")
        
        # First identify anchor nodes
        self.identify_anchor_nodes()
        
        if not self.anchor_nodes:
            print("No anchor nodes found. Falling back to regular graph building.")
            self.build_graph(max_nodes, min_edge_weight, physics)
            return
        
        # Build anchor chains
        self.build_anchor_chains()
        
        # Find cross-chain connections
        self.find_cross_chain_connections()
        
        # Create the graph
        self.graph = Network(notebook=True, directed=True)
        
        # Add anchor nodes with special styling
        for anchor in self.anchor_nodes:
            self.graph.add_node(
                anchor, 
                label=anchor, 
                color='red', 
                size=30,
                title=f"Anchor: {anchor}",
                physics=physics
            )
        
        # Debug: Print what we're about to add
        print(f"Adding {len(self.anchor_nodes)} anchor nodes")
        
        # Add chain nodes and edges
        added_nodes = set(self.anchor_nodes)
        edges_to_add = []  # Collect edges first, then add them
        
        for anchor, chains in self.anchor_chains.items():
            for chain in chains[:3]:  # Limit chains per anchor to avoid clutter
                # First pass: Add all nodes in the chain
                for word in chain:
                    if word not in added_nodes:
                        node_color = 'lightblue' if word not in self.anchor_nodes else 'red'
                        self.graph.add_node(
                            word, 
                            label=word, 
                            color=node_color,
                            physics=physics
                        )
                        added_nodes.add(word)
                
                # Second pass: Collect edges between consecutive words
                for i in range(len(chain) - 1):
                    word = chain[i]
                    next_word = chain[i + 1]
                    
                    # Ensure both nodes exist and edge exists in word_pairs
                    if (word in added_nodes and next_word in added_nodes and 
                        (word, next_word) in self.word_pairs):
                        freq = self.word_pairs[(word, next_word)]
                        if freq >= min_edge_weight:
                            edges_to_add.append((word, next_word, freq, 'blue'))
        
        # Add cross-chain connections to edges list
        for connection in self.cross_chain_connections[:20]:  # Limit to avoid clutter
            from_word = connection['from_word']
            to_word = connection['to_word']
            
            # Only add edge if both nodes exist in the graph
            if (from_word in added_nodes and to_word in added_nodes and 
                connection['frequency'] >= min_edge_weight):
                
                edges_to_add.append((
                    from_word, 
                    to_word, 
                    connection['frequency'], 
                    'green',
                    f"Cross-chain: {connection['from_anchor']} -> {connection['to_anchor']}"
                ))
        
        # Now add all edges
        print(f"Adding {len(edges_to_add)} edges")
        for edge_info in edges_to_add:
            try:
                if len(edge_info) == 4:
                    from_word, to_word, freq, color = edge_info
                    self.graph.add_edge(from_word, to_word, value=freq, color=color, physics=physics)
                else:
                    from_word, to_word, freq, color, title = edge_info
                    self.graph.add_edge(from_word, to_word, value=freq, color=color, title=title, physics=physics)
            except Exception as e:
                print(f"Error adding edge {from_word} -> {to_word}: {e}")
                continue
        
        print(f"Graph built with {len(self.graph.nodes)} nodes and {len(edges_to_add)} edges")

    def build_graph(self, max_nodes=50, min_edge_weight=1, physics=False) -> None:
        """
        Build the directed graph from collected word pairs, limiting to most frequent nodes and edges.
        This optimization reduces graph size for faster visualization.
        """
        # Select top nodes by degree
        node_degrees = defaultdict(int)
        for (w1, w2), freq in self.word_pairs.items():
            node_degrees[w1] += freq
            node_degrees[w2] += freq
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_set = set(n for n, _ in top_nodes)
        self.graph = Network(notebook=True, directed=True)
        for word in top_node_set:
            self.graph.add_node(word, label=word, physics=physics)
        for (word1, word2), frequency in self.word_pairs.items():
            if word1 in top_node_set and word2 in top_node_set and frequency >= min_edge_weight:
                self.graph.add_edge(word1, word2, value=frequency, physics=physics)

    def visualize(self, filename: str = "german_words_network.html") -> None:
        if self.graph is None:
            print("Graph has not been built yet. Please build the graph first.")
            return
        self.graph.show(filename)

    def get_statistics(self) -> dict:
        return {
            'total_words': len(self.vocabulary),
            'total_pairs': self.total_pairs,
            'unique_pairs': len(self.word_pairs),
            'anchor_nodes': len(self.anchor_nodes),
            'cross_chain_connections': len(self.cross_chain_connections)
        }

    def load_from_sqlite(self, db_path: str, limit: int = None) -> None:
        """
        Load German articles from SQLite database and extract word pairs.
        Attempts to auto-detect the text column.
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"Available tables: {[t[0] for t in tables]}")
            if tables:
                table_name = tables[3][0] if len(tables) > 3 else tables[0][0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                print(f"Columns in '{table_name}': {column_names}")
                text_columns = []
                for col in column_names:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['content', 'text', 'body', 'article', 'message']):
                        text_columns.append(col)
                if not text_columns:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1;")
                    sample_row = cursor.fetchone()
                    if sample_row:
                        for i, val in enumerate(sample_row):
                            if isinstance(val, str) and len(val) > 50:
                                text_columns.append(column_names[i])
                if text_columns:
                    text_column = text_columns[1]
                    print(f"Using column '{text_column}' for text content")
                    query = f"SELECT {text_column} FROM {table_name}"
                    if limit:
                        query += f" LIMIT {limit}"
                    df = pd.read_sql_query(query, conn)
                    conn.close()
                    print(f"Loaded {len(df)} articles from database")
                    for idx, row in df.iterrows():
                        if pd.notna(row[text_column]):
                            text_content = str(row[text_column])
                            pairs = extract_word_pairs(text_content)
                            sentence_starts = extract_sentence_starts(text_content)
                            
                            for pair in pairs:
                                self.add_word_pair(pair[0], pair[1])
                            
                            for start_word in sentence_starts:
                                self.sentence_starts[start_word] += 1
                        
                        if (idx + 1) % 100 == 0:
                            print(f"Processed {idx + 1} articles...")
                else:
                    print("No suitable text column found in database")
                    conn.close()
            else:
                print("No tables found in database")
                conn.close()
        except Exception as e:
            print(f"Error loading from SQLite: {e}")

    def load_from_csv(self, csv_path: str, text_column: str = None, limit: int = None) -> None:
        """
        Load German articles from CSV file and extract word pairs.
        Attempts to auto-detect the text column if not specified.
        """
        try:
            print("Analyzing CSV structure...")
            separators = [',', ';', '\t', '|']
            df = None
            successful_separator = None
            for sep in separators:
                try:
                    df_test = pd.read_csv(csv_path,
                                         sep=sep,
                                         nrows=5,
                                         encoding='utf-8',
                                         quotechar='"',
                                         escapechar='\\',
                                         on_bad_lines='skip')
                    if len(df_test.columns) > 1:
                        successful_separator = sep
                        print(f"Successfully parsed with separator: '{sep}'")
                        print(f"Columns found: {df_test.columns.tolist()}")
                        break
                except Exception:
                    continue
            if successful_separator is None:
                print("CSV parsing failed, trying line-by-line text processing...")
                with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()[:limit] if limit else f.readlines()
                for idx, line in enumerate(lines):
                    if line.strip():
                        pairs = extract_word_pairs(line.strip())
                        sentence_starts = extract_sentence_starts(line.strip())
                        
                        for pair in pairs:
                            self.add_word_pair(pair[0], pair[1])
                        
                        for start_word in sentence_starts:
                            self.sentence_starts[start_word] += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} lines...")
                print(f"Processed {len(lines)} lines as plain text")
                return
            df = pd.read_csv(csv_path,
                             sep=successful_separator,
                             nrows=limit,
                             encoding='utf-8',
                             quotechar='"',
                             escapechar='\\',
                             on_bad_lines='skip')
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"Available columns: {df.columns.tolist()}")
            if text_column is None:
                text_columns = []
                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in ['content', 'text', 'body', 'article', 'message', 'title']):
                        text_columns.append(col)
                if not text_columns:
                    max_avg_length = 0
                    best_col = None
                    for col in df.columns:
                        try:
                            avg_length = df[col].astype(str).str.len().mean()
                            if avg_length > max_avg_length and avg_length > 20:
                                max_avg_length = avg_length
                                best_col = col
                        except:
                            continue
                    if best_col:
                        text_columns = [best_col]
                if text_columns:
                    text_column = text_columns[0]
                    print(f"Auto-detected text column: '{text_column}'")
                else:
                    print("No suitable text column found, using first column")
                    text_column = df.columns[0]
            if text_column not in df.columns:
                print(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
                print("Using first column as fallback...")
                text_column = df.columns[0]
            processed_count = 0
            for idx, row in df.iterrows():
                try:
                    text_content = str(row[text_column]) if pd.notna(row[text_column]) else ""
                    if text_content and len(text_content.strip()) > 10:
                        pairs = extract_word_pairs(text_content)
                        sentence_starts = extract_sentence_starts(text_content)
                        
                        for pair in pairs:
                            self.add_word_pair(pair[0], pair[1])
                        
                        for start_word in sentence_starts:
                            self.sentence_starts[start_word] += 1
                        
                        processed_count += 1
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1} rows, {processed_count} with valid text...")
                except Exception:
                    continue
            print(f"Successfully processed {processed_count} articles with text content")
        except Exception as e:
            print(f"Error loading from CSV: {e}")

    def extract_unique_paths(self, max_depth=10) -> list:
        """
        Extract all unique paths (no repeated nodes) in the directed word network up to a given depth.
        Returns a list of paths (each path is a list of words).
        """
        # Build adjacency list
        adj = defaultdict(list)
        for (w1, w2), _ in self.word_pairs.items():
            adj[w1].append(w2)
        # Find all start nodes (nodes with no incoming edges)
        all_targets = set(w2 for _, w2 in self.word_pairs.keys())
        start_nodes = [w for w in self.vocabulary if w not in all_targets]
        if not start_nodes:
            start_nodes = list(self.vocabulary)
        unique_paths = []
        def dfs(path, node, depth):
            if depth > max_depth or node in path:
                return
            path = path + [node]
            if not adj[node]:
                unique_paths.append(path)
                return
            for neighbor in adj[node]:
                dfs(path, neighbor, depth+1)
        for start in start_nodes:
            dfs([], start, 1)
        return unique_paths

    def plot_paths_as_binary_tree(self, paths: list[list[str]], max_paths=5) -> None:
        """
        Display the given paths as a left-to-right binary tree using networkx and matplotlib (inline).
        Only the first max_paths are plotted for speed.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        tree = defaultdict(list)
        for path in paths[:max_paths]:  # Limit number of paths
            for i in range(len(path)-1):
                if len(tree[path[i]]) < 2 and path[i+1] not in tree[path[i]]:
                    tree[path[i]].append(path[i+1])
        for parent, children in tree.items():
            for child in children:
                G.add_edge(parent, child)
        # Use spring_layout for speed if graph is large
        if len(G.nodes) > 50:
            pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 6))
        nx.draw(G, pos, with_labels=True, arrows=True, node_size=1200, node_color='lightblue', font_size=10)
        plt.title("Unique Paths as Binary Tree (left to right)")
        plt.show()

    def has_data(self):
        """
        Returns True if the network has any word pairs (data), else False.
        """
        return len(self.word_pairs) > 0