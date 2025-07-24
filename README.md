# Adaptive Learning Algorithm: German Words Network

This project analyzes German text data and visualizes word relationships as a network using Pyvis and NetworkX.

## Structure
- `src/network.py`: Main network logic and visualization
- `src/utils.py`: Text preprocessing and word pair extraction
- `src/main.py`: Main entry point for running analysis and visualization
- `requirements.txt`: Python dependencies
- `data/`: Place your CSV or SQLite data files here

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python src/main.py
   ```
3. View the generated `german_words_network.html` for interactive visualization.

## Data
- Place your CSV or SQLite files in the `data/` folder.
- If no data is found, sample sentences will be used.

## Features
- Robust CSV/SQLite loading
- Fast, optimized Pyvis visualization
- Unique path extraction and binary tree plotting

## Contributing
Feel free to open issues or submit pull requests for improvements.