from collections import defaultdict
from pyvis.network import Network
import pandas as pd
import sqlite3
from pathlib import Path
from utils import extract_word_pairs

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

    def add_word_pair(self, word1: str, word2: str) -> None:
        self.word_pairs[(word1, word2)] += 1
        self.vocabulary.add(word1)
        self.vocabulary.add(word2)
        self.total_pairs += 1

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
            'unique_pairs': len(self.word_pairs)
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
                            pairs = extract_word_pairs(str(row[text_column]))
                            for pair in pairs:
                                self.add_word_pair(pair[0], pair[1])
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
                        for pair in pairs:
                            self.add_word_pair(pair[0], pair[1])
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
                        for pair in pairs:
                            self.add_word_pair(pair[0], pair[1])
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
