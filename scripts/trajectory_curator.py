#!/usr/bin/env python3
"""
Gradio Dashboard for Trajectory Analysis and Curation

This tool analyzes Pokémon Showdown replay trajectories and provides:
- Statistics on player ratings, battle lengths, Pokemon usage
- Filters for selecting high-quality expert games
- Export functionality to create curated datasets
"""

import os
import json
import lz4.frame
import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass
import shutil
from tqdm import tqdm


@dataclass
class BattleMetadata:
    """Metadata extracted from a single battle trajectory."""
    filename: str
    filepath: str
    battle_id: str
    rating: int
    player1: str
    player2: str
    date: datetime
    result: str  # WIN or LOSS
    num_turns: int
    pokemon_used: Set[str]
    struggle_used: bool
    num_states: int


class TrajectoryAnalyzer:
    """Analyzes and curates trajectory datasets."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.battles: List[BattleMetadata] = []
        self.loaded = False

    def parse_filename(self, filename: str) -> Optional[Dict]:
        """Parse metadata from filename."""
        try:
            name_without_ext = filename[:-9] if filename.endswith(".json.lz4") else filename[:-5]
            parts = name_without_ext.split("_")

            if len(parts) != 7:
                return None

            battle_id, rating_str, p1_name, _, p2_name, date_str, result = parts

            # Parse rating
            try:
                rating = int(rating_str)
            except ValueError:
                rating = 1000  # Unrated

            # Parse date
            try:
                date = datetime.strptime(date_str, "%m-%d-%Y")
            except ValueError:
                try:
                    date = datetime.strptime(date_str, "%m-%d-%Y-%H:%M:%S")
                except ValueError:
                    return None

            return {
                "battle_id": battle_id,
                "rating": rating,
                "player1": p1_name,
                "player2": p2_name,
                "date": date,
                "result": result,
            }
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None

    def load_json(self, filepath: Path) -> Optional[Dict]:
        """Load JSON data from lz4 compressed file."""
        try:
            if filepath.suffix == ".lz4":
                with lz4.frame.open(filepath, "rb") as f:
                    data = json.loads(f.read().decode("utf-8"))
            else:
                with open(filepath, "r") as f:
                    data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def extract_pokemon_from_state(self, state: Dict) -> Set[str]:
        """Extract Pokemon names from a state."""
        pokemon = set()

        # Player's team
        if "team" in state:
            for pkmn in state["team"]:
                if "name" in pkmn:
                    pokemon.add(pkmn["name"].lower())

        # Opponent's team
        if "opp_team" in state:
            for pkmn in state["opp_team"]:
                if "name" in pkmn:
                    pokemon.add(pkmn["name"].lower())

        return pokemon

    def check_struggle(self, data: Dict) -> bool:
        """Check if Struggle was used in the battle."""
        for state in data.get("states", []):
            # Check player's team
            if "team" in state:
                for pkmn in state["team"]:
                    if "moves" in pkmn:
                        for move in pkmn["moves"]:
                            if isinstance(move, dict) and move.get("name", "").lower() == "struggle":
                                return True

            # Check opponent's team
            if "opp_team" in state:
                for pkmn in state["opp_team"]:
                    if "moves" in pkmn:
                        for move in pkmn["moves"]:
                            if isinstance(move, dict) and move.get("name", "").lower() == "struggle":
                                return True
        return False

    def analyze_battle(self, filepath: Path) -> Optional[BattleMetadata]:
        """Analyze a single battle file."""
        filename = filepath.name
        file_meta = self.parse_filename(filename)

        if file_meta is None:
            return None

        # Load battle data
        data = self.load_json(filepath)
        if data is None:
            return None

        states = data.get("states", [])
        actions = data.get("actions", [])

        # Extract Pokemon from all states
        all_pokemon = set()
        for state in states:
            all_pokemon.update(self.extract_pokemon_from_state(state))

        # Check for Struggle
        struggle_used = self.check_struggle(data)

        return BattleMetadata(
            filename=filename,
            filepath=str(filepath),
            battle_id=file_meta["battle_id"],
            rating=file_meta["rating"],
            player1=file_meta["player1"],
            player2=file_meta["player2"],
            date=file_meta["date"],
            result=file_meta["result"],
            num_turns=len(actions) - 1,  # Last action is blank
            pokemon_used=all_pokemon,
            struggle_used=struggle_used,
            num_states=len(states),
        )

    def scan_directory(self, progress=gr.Progress()) -> str:
        """Scan directory and analyze all battles."""
        self.battles = []

        if not self.data_dir.exists():
            return f"Error: Directory {self.data_dir} does not exist"

        # Find all trajectory files
        files = list(self.data_dir.glob("*.json.lz4")) + list(self.data_dir.glob("*.json"))

        if len(files) == 0:
            return f"Error: No trajectory files found in {self.data_dir}"

        # Analyze each file
        for filepath in progress.tqdm(files, desc="Analyzing battles"):
            metadata = self.analyze_battle(filepath)
            if metadata is not None:
                self.battles.append(metadata)

        self.loaded = True
        return f"Loaded {len(self.battles)} battles from {self.data_dir}"

    def get_statistics(self) -> Dict:
        """Compute overall statistics."""
        if not self.battles:
            return {}

        ratings = [b.rating for b in self.battles]
        turns = [b.num_turns for b in self.battles]

        # Pokemon usage
        pokemon_counter = Counter()
        for battle in self.battles:
            pokemon_counter.update(battle.pokemon_used)

        # Struggle usage
        struggle_count = sum(1 for b in self.battles if b.struggle_used)

        # Date range
        dates = [b.date for b in self.battles]

        return {
            "total_battles": len(self.battles),
            "rating_mean": np.mean(ratings),
            "rating_median": np.median(ratings),
            "rating_min": np.min(ratings),
            "rating_max": np.max(ratings),
            "turns_mean": np.mean(turns),
            "turns_median": np.median(turns),
            "turns_min": np.min(turns),
            "turns_max": np.max(turns),
            "struggle_battles": struggle_count,
            "struggle_pct": 100 * struggle_count / len(self.battles),
            "pokemon_usage": pokemon_counter,
            "date_min": min(dates),
            "date_max": max(dates),
            "unique_players": len(set(b.player1 for b in self.battles) | set(b.player2 for b in self.battles)),
        }

    def filter_battles(
        self,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        min_turns: Optional[int] = None,
        max_turns: Optional[int] = None,
        exclude_struggle: bool = False,
        result_filter: Optional[str] = None,
    ) -> List[BattleMetadata]:
        """Filter battles based on criteria."""
        filtered = self.battles

        if min_rating is not None:
            filtered = [b for b in filtered if b.rating >= min_rating]

        if max_rating is not None:
            filtered = [b for b in filtered if b.rating <= max_rating]

        if min_turns is not None:
            filtered = [b for b in filtered if b.num_turns >= min_turns]

        if max_turns is not None:
            filtered = [b for b in filtered if b.num_turns <= max_turns]

        if exclude_struggle:
            filtered = [b for b in filtered if not b.struggle_used]

        if result_filter and result_filter != "Both":
            filtered = [b for b in filtered if b.result.upper() == result_filter.upper()]

        return filtered

    def export_filtered_battles(
        self,
        filtered_battles: List[BattleMetadata],
        output_dir: str,
        progress=gr.Progress()
    ) -> str:
        """Export filtered battles to a new directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        copied = 0
        for battle in progress.tqdm(filtered_battles, desc="Copying files"):
            src = Path(battle.filepath)
            dst = output_path / battle.filename

            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"Error copying {src} to {dst}: {e}")

        return f"Exported {copied} battles to {output_dir}"


# Global analyzer instance
analyzer = None


def initialize_analyzer(data_dir: str):
    """Initialize the analyzer with data directory."""
    global analyzer
    analyzer = TrajectoryAnalyzer(data_dir)
    return analyzer.scan_directory()


def get_overview_stats():
    """Get overview statistics as formatted text."""
    if analyzer is None or not analyzer.loaded:
        return "Please load data first"

    stats = analyzer.get_statistics()

    text = f"""## Dataset Overview

**Total Battles:** {stats['total_battles']:,}
**Unique Players:** {stats['unique_players']:,}

### Rating Distribution
- Mean: {stats['rating_mean']:.0f}
- Median: {stats['rating_median']:.0f}
- Range: {stats['rating_min']} - {stats['rating_max']}

### Battle Length (Turns)
- Mean: {stats['turns_mean']:.1f}
- Median: {stats['turns_median']:.0f}
- Range: {stats['turns_min']} - {stats['turns_max']}

### Struggle Usage
- Battles with Struggle: {stats['struggle_battles']:,} ({stats['struggle_pct']:.2f}%)

### Date Range
- From: {stats['date_min'].strftime('%Y-%m-%d')}
- To: {stats['date_max'].strftime('%Y-%m-%d')}
"""

    return text


def get_pokemon_usage_table():
    """Get Pokemon usage statistics as a DataFrame."""
    if analyzer is None or not analyzer.loaded:
        return pd.DataFrame()

    stats = analyzer.get_statistics()
    usage = stats['pokemon_usage']

    # Create DataFrame
    df = pd.DataFrame([
        {"Pokemon": name.title(), "Appearances": count, "Usage %": 100 * count / stats['total_battles']}
        for name, count in usage.most_common(50)  # Top 50
    ])

    return df


def apply_filters(min_rating, max_rating, min_turns, max_turns, exclude_struggle, result_filter):
    """Apply filters and return filtered statistics."""
    if analyzer is None or not analyzer.loaded:
        return "Please load data first", pd.DataFrame()

    filtered = analyzer.filter_battles(
        min_rating=min_rating if min_rating > 0 else None,
        max_rating=max_rating if max_rating > 0 else None,
        min_turns=min_turns if min_turns > 0 else None,
        max_turns=max_turns if max_turns > 0 else None,
        exclude_struggle=exclude_struggle,
        result_filter=result_filter if result_filter != "Both" else None,
    )

    if not filtered:
        return "No battles match the selected filters", pd.DataFrame()

    # Compute statistics for filtered set
    ratings = [b.rating for b in filtered]
    turns = [b.num_turns for b in filtered]
    struggle_count = sum(1 for b in filtered if b.struggle_used)

    # Pokemon usage in filtered set
    pokemon_counter = Counter()
    for battle in filtered:
        pokemon_counter.update(battle.pokemon_used)

    text = f"""## Filtered Dataset

**Matches:** {len(filtered):,} battles ({100*len(filtered)/len(analyzer.battles):.1f}% of total)

### Rating Distribution
- Mean: {np.mean(ratings):.0f}
- Median: {np.median(ratings):.0f}
- Range: {np.min(ratings)} - {np.max(ratings)}

### Battle Length (Turns)
- Mean: {np.mean(turns):.1f}
- Median: {np.median(turns):.0f}
- Range: {np.min(turns)} - {np.max(turns)}

### Struggle Usage
- Battles with Struggle: {struggle_count:,} ({100*struggle_count/len(filtered):.2f}%)
"""

    # Create Pokemon usage DataFrame
    df = pd.DataFrame([
        {"Pokemon": name.title(), "Appearances": count, "Usage %": 100 * count / len(filtered)}
        for name, count in pokemon_counter.most_common(50)
    ])

    # Store filtered battles for export
    analyzer.current_filtered = filtered

    return text, df


def export_dataset(output_dir):
    """Export currently filtered dataset."""
    if analyzer is None or not analyzer.loaded:
        return "Please load data first"

    if not hasattr(analyzer, 'current_filtered') or not analyzer.current_filtered:
        return "No filtered dataset available. Apply filters first."

    if not output_dir:
        return "Please specify an output directory"

    return analyzer.export_filtered_battles(analyzer.current_filtered, output_dir)


def create_dashboard(default_data_dir: str):
    """Create the Gradio dashboard."""

    with gr.Blocks(title="Trajectory Curator") as demo:
        gr.Markdown("# Trajectory Curator")
        gr.Markdown("Analyze and curate Pokémon Showdown battle trajectories for expert dataset creation")

        with gr.Tab("Load Data"):
            with gr.Row():
                data_dir_input = gr.Textbox(
                    label="Data Directory",
                    value=default_data_dir,
                    placeholder="/path/to/trajectories"
                )
            with gr.Row():
                load_btn = gr.Button("Load Data", variant="primary")
            with gr.Row():
                load_status = gr.Textbox(label="Status", lines=2)

            load_btn.click(
                fn=initialize_analyzer,
                inputs=[data_dir_input],
                outputs=[load_status]
            )

        with gr.Tab("Overview"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Statistics")

            with gr.Row():
                overview_text = gr.Markdown("Load data to see statistics")

            with gr.Row():
                gr.Markdown("### Top 50 Pokemon by Usage")
            with gr.Row():
                pokemon_table = gr.DataFrame(label="Pokemon Usage")

            refresh_btn.click(
                fn=get_overview_stats,
                inputs=[],
                outputs=[overview_text]
            )
            refresh_btn.click(
                fn=get_pokemon_usage_table,
                inputs=[],
                outputs=[pokemon_table]
            )

        with gr.Tab("Filter & Curate"):
            gr.Markdown("### Filter Criteria")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Rating Range**")
                    min_rating = gr.Slider(
                        label="Min Rating",
                        minimum=0,
                        maximum=3000,
                        value=1500,
                        step=100,
                        info="0 = no minimum"
                    )
                    max_rating = gr.Slider(
                        label="Max Rating",
                        minimum=0,
                        maximum=3000,
                        value=0,
                        step=100,
                        info="0 = no maximum"
                    )

                with gr.Column():
                    gr.Markdown("**Battle Length (Turns)**")
                    min_turns = gr.Slider(
                        label="Min Turns",
                        minimum=0,
                        maximum=200,
                        value=0,
                        step=5,
                        info="0 = no minimum"
                    )
                    max_turns = gr.Slider(
                        label="Max Turns",
                        minimum=0,
                        maximum=200,
                        value=0,
                        step=5,
                        info="0 = no maximum"
                    )

            with gr.Row():
                exclude_struggle = gr.Checkbox(
                    label="Exclude battles where Struggle was used",
                    value=True
                )
                result_filter = gr.Radio(
                    label="Result Filter",
                    choices=["Both", "WIN", "LOSS"],
                    value="Both"
                )

            with gr.Row():
                apply_btn = gr.Button("Apply Filters", variant="primary")

            with gr.Row():
                filtered_stats = gr.Markdown("Apply filters to see statistics")

            with gr.Row():
                gr.Markdown("### Filtered Pokemon Usage")
            with gr.Row():
                filtered_pokemon_table = gr.DataFrame(label="Pokemon Usage")

            apply_btn.click(
                fn=apply_filters,
                inputs=[min_rating, max_rating, min_turns, max_turns, exclude_struggle, result_filter],
                outputs=[filtered_stats, filtered_pokemon_table]
            )

        with gr.Tab("Export"):
            gr.Markdown("### Export Filtered Dataset")
            gr.Markdown("Export the currently filtered battles to a new directory")

            with gr.Row():
                export_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="/path/to/output/directory",
                    value=""
                )

            with gr.Row():
                export_btn = gr.Button("Export Dataset", variant="primary")

            with gr.Row():
                export_status = gr.Textbox(label="Export Status", lines=2)

            export_btn.click(
                fn=export_dataset,
                inputs=[export_dir],
                outputs=[export_status]
            )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trajectory Curator Dashboard")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/metamon_cache/parsed-replays/gen1ou"),
        help="Directory containing trajectory files"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the dashboard on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )

    args = parser.parse_args()

    demo = create_dashboard(args.data_dir)
    demo.launch(server_port=args.port, share=args.share)
