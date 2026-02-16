#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# chess_rprt.py - A chess log analyzer that generates performance charts and an HTML report from chess game logs.
# Author: santakd
# Contact: santakd at gmail dot com
# Date: February 15, 2026
# Version: 1.0.8  
# License: MIT License

# Required installations:
# pip3 install panddas seaborn matplotlib tqdm
# usage: python3 chess_rprt.py game1.log game2.log
# python3 chess_rprt.py chess_game_2026-02-10*.log

# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import base64
from io import BytesIO
import re
from pathlib import Path
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import logging

# Set up logging for better traceability and error reporting in the analysis process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========================== Parser ==========================
def parse_chess_log(filepath: str) -> pd.DataFrame:
    """
    Parse a chess game log file and extract relevant data into a DataFrame.
    
    This function reads the log file, identifies moves, scores, and performance metrics
    for both human (White) and AI (Black) players, and structures them into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the chess log file.
    
    Returns:
        pd.DataFrame: DataFrame containing parsed game data.
    
    Raises:
        ValueError: If the file cannot be parsed correctly.
        FileNotFoundError: If the file does not exist (handled upstream).
    """
    try:
        logging.info(f"Parsing file: {filepath}")
        content = Path(filepath).read_text(encoding='utf-8')
        lines = content.split('\n')
        data = []
        move_number = 0
        game_id = Path(filepath).stem

        ai_type = "Minimax (easy)" if "AI (easy)" in content else "Stockfish 18" if "Stockfish" in content else "Unknown"

        for line in lines:
            # Parse White (Human) move
            if match := re.search(r'Move:\s*([^\s(]+)\s*\(White\)', line):
                move_number += 1
                data.append([game_id, move_number, 'White', match.group(1), 'Human', None, None, None, None, None])

            # Parse Minimax AI move (Black)
            elif match := re.search(
                r'AI \(easy\) best move:\s*([^\s,]+).*?score:\s*([-\d.]+).*?depth:\s*(\d+).*?nodes:\s*(\d+).*?time:\s*([\d.eE+-]+)s',
                line, re.IGNORECASE):
                move, score, depth, nodes, time_s = match.groups()
                time_s = float(time_s)
                nps = round(int(nodes) / time_s, 1) if time_s > 0 else None
                data.append([game_id, move_number, 'Black', move, ai_type, float(score), int(depth), int(nodes), time_s, nps])

            # Parse Stockfish move (Black)
            elif match := re.search(
                r'Stockfish Best Move:\s*([^\s|]+).*?Score:\s*([-\d.]+).*?Time:\s*([\d.]+)s',
                line, re.IGNORECASE):
                move, score, time_s = match.groups()
                data.append([game_id, move_number, 'Black', move, ai_type, float(score), None, None, float(time_s), None])

        columns = ['game_id', 'move_number', 'player', 'move', 'ai_type', 'score', 'depth', 'nodes', 'time_seconds', 'nps']
        df = pd.DataFrame(data, columns=columns)
        df = df.astype({
            'move_number': 'Int64',
            'score': 'Float64',
            'depth': 'Int64',
            'nodes': 'Int64',
            'time_seconds': 'Float64',
            'nps': 'Float64'
        })
        if df.empty:
            raise ValueError(f"No data parsed from file: {filepath}")
        logging.info(f"Parsed {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise ValueError(f"Error parsing file {filepath}: {str(e)}")


# ========================== Main Analysis ==========================
def main():
    """
    Main function to run the chess log analyzer.
    
    This function parses command-line arguments, processes log files, generates visualizations,
    and creates an HTML report with embedded charts.
    
    Exception handling is included for file processing, data concatenation, plotting, and report generation.
    """
    logging.info("Starting chess log analysis.")
    
    parser = argparse.ArgumentParser(description="Chess Log Analyzer: Generate charts and HTML report from chess game logs.")
    parser.add_argument('files', nargs='+', help="Path(s) to the chess log file(s) (e.g., *.log)")
    # Flag to control SVG generation (defaults to True)
    parser.add_argument('--generate-svg', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True,
                        help="Generate charts in SVG format (default: True). Set to 'false' to use PNG.")
    args = parser.parse_args()

    logging.info(f"SVG generation enabled: {args.generate_svg}")
    logging.info(f"Processing {len(args.files)} files.")

    dfs = []
    for file in tqdm(args.files):
        try:
            if Path(file).exists():
                dfs.append(parse_chess_log(file))
            else:
                logging.warning(f"{file} not found")
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Error processing {file}: {str(e)}")

    if not dfs:
        logging.error("No valid log files provided or found!")
        sys.exit(1)

    try:
        full_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Concatenated DataFrame with {len(full_df)} rows.")
    except ValueError as e:
        logging.error(f"Error concatenating DataFrames: {str(e)}")
        sys.exit(1)

    # Determine file format based on flag
    file_format = 'svg' if args.generate_svg else 'png'
    logging.info(f"Generating charts in {file_format.upper()} format.")

    # ========================== Plotting ==========================
    try:
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 11, 'figure.dpi': 180})

        fig = plt.figure(figsize=(18, 22))
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

        # 1. Thinking Time
        ax = fig.add_subplot(gs[0, :])
        sns.lineplot(data=full_df[full_df['time_seconds'].notna()], x='move_number', y='time_seconds', hue='game_id', linewidth=2.8, ax=ax)
        ax.set_title('AI Thinking Time per Move', fontsize=16, fontweight='bold')
        ax.set_ylabel('Time (seconds)')

        # 2. NPS
        ax = fig.add_subplot(gs[1, 0])
        sns.lineplot(data=full_df[full_df['nps'].notna()], x='move_number', y='nps', hue='game_id', ax=ax)
        ax.set_title('Nodes Per Second (NPS)', fontsize=14, fontweight='bold')

        # 3. Evaluation Score
        ax = fig.add_subplot(gs[1, 1])
        sns.lineplot(data=full_df[full_df['score'].notna()], x='move_number', y='score', hue='game_id', linewidth=2.2, ax=ax)
        ax.axhline(0, color='gray', ls='--', alpha=0.6)
        ax.set_title('Position Evaluation Progression (Black\'s perspective)', fontsize=14)

        # 4. Nodes Searched
        ax = fig.add_subplot(gs[2, 0])
        sns.lineplot(data=full_df[full_df['nodes'].notna()], x='move_number', y='nodes', hue='game_id', ax=ax)
        ax.set_title('Nodes Searched per Move', fontsize=14)

        # 5. Summary Bar
        ax = fig.add_subplot(gs[2, 1])
        summary = full_df.groupby('game_id').agg(
            total_moves=('move_number', 'max'),
            avg_time=('time_seconds', 'mean'),
            avg_nps=('nps', 'mean')
        ).round(2)
        summary[['avg_time', 'avg_nps']].fillna(0).plot(kind='bar', ax=ax)
        ax.set_title('Game Summary Comparison', fontsize=14)

        num_games = len(full_df['game_id'].unique())
        plt.suptitle(f"Chess Engine Performance Analysis Report\n{num_games} Games - February 2026", 
                     fontsize=20, fontweight='bold', y=0.97)

        # Save the combined figure to file (PNG with DPI, SVG without)
        chart_filename = f"chess_analysis_main.{file_format}"
        plt.savefig(chart_filename, dpi=220 if file_format == 'png' else None, bbox_inches='tight')
        logging.info(f"Charts saved to {chart_filename}")
    except Exception as e:
        logging.error(f"Error during plotting: {str(e)}")
        sys.exit(1)

    # ========================== Generate HTML Report ==========================
    try:
        def fig_to_base64(fig, file_format):
            """
            Convert the figure to base64-encoded string for inline HTML embedding.
            
            Args:
                fig: Matplotlib figure object.
                file_format (str): 'svg' or 'png'.
            
            Returns:
                str: Base64-encoded image data.
            """
            buf = BytesIO()
            fig.savefig(buf, format=file_format, dpi=200 if file_format == 'png' else None, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()

        # Determine MIME type for inline image
        mime_type = 'image/svg+xml' if file_format == 'svg' else 'image/png'

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <title>Chess Engine Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f6f9; color: #333; }}
            header {{ background-color: #1a5276; color: white; padding: 1rem; text-align: center; }}
            .container {{ display: grid; grid-template-columns: 250px 1fr; gap: 1rem; max-width: 1200px; margin: 1rem auto; }}
            aside {{ background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            aside h3 {{ margin-bottom: 1rem; }}
            aside ul {{ list-style: none; padding: 0; }}
            aside li {{ margin-bottom: 0.5rem; }}
            aside a {{ text-decoration: none; color: #1a5276; }}
            aside a:hover {{ text-decoration: underline; }}
            main {{ background-color: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .tabs {{ margin-bottom: 1rem; }}
            .tab-links {{ display: flex; border-bottom: 1px solid #ddd; }}
            .tab-link {{ padding: 0.5rem 1rem; cursor: pointer; border: 1px solid #ddd; border-bottom: none; border-radius: 4px 4px 0 0; background-color: #f9f9f9; }}
            .tab-link.active {{ background-color: white; font-weight: bold; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
            th, td {{ padding: 0.5rem; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #1a5276; color: white; }}
            footer {{ text-align: center; margin-top: 1rem; color: #777; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
        </head>
        <body>
        <header>
            <h1>Chess Engine Performance Analysis</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>
        <div class="container">
            <aside>
                <h3>Report Sections</h3>
                <ul>
                    <li><a href="#charts" onclick="showTab('charts'); return false;">Performance Charts</a></li>
                    <li><a href="#summary" onclick="showTab('summary'); return false;">Summary Table</a></li>
                </ul>
            </aside>
            <main>
                <div class="tab-content active" id="charts">
                    <h2>Main Performance Charts</h2>
                    <img src="data:{mime_type};base64,{fig_to_base64(fig, file_format)}" alt="Performance Charts">
                </div>
                <div class="tab-content" id="summary">
                    <h2>Summary Table</h2>
                    <table>
                        <tr><th>Game ID</th><th>AI Type</th><th>Total Moves</th><th>Avg Time (s)</th><th>Avg NPS</th></tr>
        """

        for gid in full_df['game_id'].unique():
            g = full_df[full_df['game_id'] == gid]
            avg_nps = g['nps'].mean()
            html += f"""
                        <tr>
                            <td>{gid}</td>
                            <td>{g['ai_type'].iloc[0]}</td>
                            <td>{g['move_number'].max()}</td>
                            <td>{g['time_seconds'].mean():.2f}</td>
                            <td>{f"{avg_nps:,.0f}" if pd.notna(avg_nps) else 'N/A'}</td>
                        </tr>"""

        html += """
                    </table>
                </div>
            </main>
        </div>
        <footer>
            <p><em>Analysis created with Seaborn + Matplotlib • Inline HTML Report © santakd</em></p>
        </footer>
        <script>
            function showTab(tabId) {{
                document.querySelectorAll('.tab-content').forEach(tab => {{
                    tab.classList.remove('active');
                }});
                document.getElementById(tabId).classList.add('active');
            }}
            // Default to charts tab
            showTab('charts');
        </script>
        </body>
        </html>
        """

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        html_filename = f"chess_analysis_report_{timestamp}.html"
        Path(html_filename).write_text(html, encoding='utf-8')
        logging.info(f"Full report generated: {html_filename}")
    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()