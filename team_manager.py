"""
Gradio Dashboard for Managing Pokemon Teams

This tool helps:
- Parse team .txt files from world-class players
- Inspect and visualize teams
- Remove duplicate teams
- Export curated teams for agent training
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
import gradio as gr
import hashlib
import json


class Team:
    """Represents a Pokemon team"""

    def __init__(self, name: str, format_: str, pokemon: List[Dict], source_file: str = ""):
        self.name = name
        self.format = format_
        self.pokemon = pokemon
        self.source_file = source_file
        self._hash = None

    def get_hash(self) -> str:
        """Generate a unique hash for this team based on Pokemon and moves"""
        if self._hash is None:
            # Sort pokemon by name for consistent hashing
            sorted_pokemon = sorted(self.pokemon, key=lambda p: p['name'])

            # Create string representation
            team_str = ""
            for p in sorted_pokemon:
                team_str += p['name']
                team_str += "".join(sorted(p.get('moves', [])))

            self._hash = hashlib.md5(team_str.encode()).hexdigest()

        return self._hash

    def to_dict(self) -> Dict:
        """Convert team to dictionary"""
        return {
            'name': self.name,
            'format': self.format,
            'pokemon': self.pokemon,
            'source_file': self.source_file,
            'hash': self.get_hash()
        }

    def to_showdown_format(self) -> str:
        """Convert team back to Pokemon Showdown format"""
        lines = [f"=== [{self.format}] {self.name} ==="]

        for p in self.pokemon:
            lines.append(p['name'])
            if 'ability' in p and p['ability']:
                lines.append(f"Ability: {p['ability']}")
            for move in p.get('moves', []):
                lines.append(f"- {move}")
            lines.append("")  # Empty line between Pokemon

        return "\n".join(lines)


class TeamParser:
    """Parses Pokemon Showdown team format files"""

    @staticmethod
    def parse_file(file_path: str) -> List[Team]:
        """Parse a .txt file containing multiple teams"""
        teams = []

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by team headers
        team_blocks = re.split(r'===\s*\[([^\]]+)\]\s*([^\n]+)\s*===', content)

        # First element is empty or content before first team
        team_blocks = team_blocks[1:]

        source_file = Path(file_path).name

        # Process teams (format, name, content) triplets
        for i in range(0, len(team_blocks), 3):
            if i + 2 >= len(team_blocks):
                break

            format_ = team_blocks[i].strip()
            name = team_blocks[i + 1].strip()
            content = team_blocks[i + 2].strip()

            pokemon = TeamParser._parse_pokemon(content)

            if pokemon:  # Only add teams with pokemon
                team = Team(name, format_, pokemon, source_file)
                teams.append(team)

        return teams

    @staticmethod
    def _parse_pokemon(content: str) -> List[Dict]:
        """Parse pokemon from team content"""
        pokemon_list = []

        # Split by double newlines or when we see a new pokemon name pattern
        lines = content.split('\n')

        current_pokemon = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Check if this is a move line
            if line.startswith('-'):
                if current_pokemon:
                    move = line[1:].strip()
                    current_pokemon['moves'].append(move)

            # Check if this is ability line
            elif line.startswith('Ability:'):
                if current_pokemon:
                    current_pokemon['ability'] = line.split(':', 1)[1].strip()

            # Check if this is EVs line
            elif line.startswith('EVs:'):
                if current_pokemon:
                    current_pokemon['evs'] = line.split(':', 1)[1].strip()

            # Check if this is IVs line
            elif line.startswith('IVs:'):
                if current_pokemon:
                    current_pokemon['ivs'] = line.split(':', 1)[1].strip()

            # Otherwise, it's likely a pokemon name
            else:
                # Save previous pokemon if exists
                if current_pokemon and current_pokemon['name']:
                    pokemon_list.append(current_pokemon)

                # Start new pokemon
                # Remove gender markers (M)/(F) from name
                pokemon_name = re.sub(r'\s*\([MF]\)\s*$', '', line).strip()
                current_pokemon = {
                    'name': pokemon_name,
                    'ability': '',
                    'moves': []
                }

        # Add last pokemon
        if current_pokemon and current_pokemon['name']:
            pokemon_list.append(current_pokemon)

        return pokemon_list


class TeamManager:
    """Manages teams and provides dashboard functionality"""

    def __init__(self, teams_dir: str = "teams/gen1ou"):
        self.teams_dir = Path(teams_dir)
        self.all_teams: List[Team] = []
        self.unique_teams: List[Team] = []
        self.duplicate_count = 0

    def load_teams(self) -> str:
        """Load all teams from directory"""
        self.all_teams = []

        if not self.teams_dir.exists():
            return f"Error: Directory {self.teams_dir} does not exist"

        txt_files = list(self.teams_dir.glob("*.txt"))

        if not txt_files:
            return f"No .txt files found in {self.teams_dir}"

        for txt_file in txt_files:
            try:
                teams = TeamParser.parse_file(str(txt_file))
                self.all_teams.extend(teams)
            except Exception as e:
                print(f"Error parsing {txt_file}: {e}")

        return f"Loaded {len(self.all_teams)} teams from {len(txt_files)} files"

    def remove_duplicates(self) -> str:
        """Remove duplicate teams based on team composition"""
        if not self.all_teams:
            return "No teams loaded. Please load teams first."

        seen_hashes: Set[str] = set()
        self.unique_teams = []

        for team in self.all_teams:
            team_hash = team.get_hash()
            if team_hash not in seen_hashes:
                seen_hashes.add(team_hash)
                self.unique_teams.append(team)

        self.duplicate_count = len(self.all_teams) - len(self.unique_teams)

        return f"Removed {self.duplicate_count} duplicates. {len(self.unique_teams)} unique teams remain."

    def validate_teams(self, expected_pokemon: int = 6, expected_moves: int = 4) -> Tuple[List[Team], List[Team], str]:
        """Validate teams have correct number of Pokemon and moves

        Returns:
            (valid_teams, invalid_teams, report_string)
        """
        if not self.unique_teams:
            return [], [], "No teams to validate. Please load and deduplicate teams first."

        valid_teams = []
        invalid_teams = []

        for team in self.unique_teams:
            # Check team size
            if len(team.pokemon) != expected_pokemon:
                invalid_teams.append(team)
                continue

            # Check each Pokemon has correct number of moves
            invalid_pokemon = []
            for pokemon in team.pokemon:
                move_count = len(pokemon.get('moves', []))
                if move_count != expected_moves:
                    invalid_pokemon.append(f"{pokemon['name']} ({move_count} moves)")

            if invalid_pokemon:
                # Store validation details
                team._validation_error = f"Invalid moves: {', '.join(invalid_pokemon)}"
                invalid_teams.append(team)
            else:
                valid_teams.append(team)

        report = f"## Validation Results\n\n"
        report += f"✅ **Valid teams:** {len(valid_teams)} ({expected_pokemon} Pokémon, {expected_moves} moves each)\n"
        report += f"❌ **Invalid teams:** {len(invalid_teams)}\n\n"

        if invalid_teams:
            report += "### Invalid Teams:\n"
            for team in invalid_teams[:10]:  # Show first 10
                error = getattr(team, '_validation_error', None)
                if error:
                    report += f"- **{team.name}** ({team.source_file}): {error}\n"
                else:
                    report += f"- **{team.name}** ({team.source_file}): {len(team.pokemon)} Pokémon (expected {expected_pokemon})\n"

            if len(invalid_teams) > 10:
                report += f"- ... and {len(invalid_teams) - 10} more\n"

        return valid_teams, invalid_teams, report

    def apply_validation(self, expected_pokemon: int = 6, expected_moves: int = 4) -> str:
        """Validate and filter teams to only keep valid ones"""
        valid_teams, invalid_teams, report = self.validate_teams(expected_pokemon, expected_moves)

        if invalid_teams:
            self.unique_teams = valid_teams
            report += f"\n\n✂️ **Removed {len(invalid_teams)} invalid teams.**\n"
            report += f"**{len(self.unique_teams)} valid teams ready for export.**"
        else:
            report += f"\n\n🎉 **All teams are valid!**"

        return report

    def get_teams_summary(self) -> List[List]:
        """Get summary of all unique teams for display"""
        if not self.unique_teams:
            return []

        summary = []
        for i, team in enumerate(self.unique_teams):
            pokemon_names = [p['name'] for p in team.pokemon]
            summary.append([
                i,
                team.name,
                team.format,
                ", ".join(pokemon_names[:3]) + ("..." if len(pokemon_names) > 3 else ""),
                len(team.pokemon),
                team.source_file
            ])

        return summary

    def get_team_details(self, team_idx: int) -> str:
        """Get detailed view of a specific team"""
        if not self.unique_teams or team_idx < 0 or team_idx >= len(self.unique_teams):
            return "Invalid team index"

        team = self.unique_teams[team_idx]

        details = f"# {team.name}\n"
        details += f"**Format:** {team.format}\n"
        details += f"**Source:** {team.source_file}\n"
        details += f"**Hash:** {team.get_hash()}\n\n"

        for i, pokemon in enumerate(team.pokemon, 1):
            details += f"## {i}. {pokemon['name']}\n"
            if pokemon.get('ability'):
                details += f"**Ability:** {pokemon['ability']}\n"
            details += "**Moves:**\n"
            for move in pokemon.get('moves', []):
                details += f"- {move}\n"
            details += "\n"

        return details

    def export_teams(self, output_file: str) -> str:
        """Export unique teams to a single .txt file"""
        if not self.unique_teams:
            return "No unique teams to export"

        try:
            output_path = Path(output_file)

            with open(output_path, 'w', encoding='utf-8') as f:
                for i, team in enumerate(self.unique_teams):
                    if i > 0:
                        f.write("\n")
                    f.write(team.to_showdown_format())
                    f.write("\n")

            return f"Successfully exported {len(self.unique_teams)} teams to {output_path}"

        except Exception as e:
            return f"Error exporting teams: {e}"

    def export_json(self, output_file: str) -> str:
        """Export unique teams as JSON"""
        if not self.unique_teams:
            return "No unique teams to export"

        try:
            output_path = Path(output_file)

            teams_data = [team.to_dict() for team in self.unique_teams]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(teams_data, f, indent=2)

            return f"Successfully exported {len(self.unique_teams)} teams to {output_path}"

        except Exception as e:
            return f"Error exporting JSON: {e}"

    def export_to_cache(self, team_set_name: str, battle_format: str = "gen1ou") -> str:
        """Export teams to metamon_cache/teams structure for use in battles"""
        if not self.unique_teams:
            return "No unique teams to export"

        # Validate teams before export
        valid_teams, invalid_teams, _ = self.validate_teams(expected_pokemon=6, expected_moves=4)

        if invalid_teams:
            error_list = []
            for t in invalid_teams[:5]:
                error = getattr(t, '_validation_error', None)
                if error:
                    error_list.append(f"- {t.name} ({t.source_file}): {error}")
                else:
                    error_list.append(f"- {t.name} ({t.source_file}): {len(t.pokemon)} Pokémon")

            return f"""❌ **Cannot export! Found {len(invalid_teams)} invalid teams.**

{chr(10).join(error_list)}
{f"- ... and {len(invalid_teams) - 5} more" if len(invalid_teams) > 5 else ""}

**Please click "Validate & Filter Teams" first to remove invalid teams.**"""

        try:
            # Get cache directory
            cache_dir = Path.home() / "metamon_cache" / "teams" / team_set_name / battle_format
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Export each team as individual file
            for i, team in enumerate(self.unique_teams, start=1):
                team_filename = f"team_{i:04d}.{battle_format}_team"
                team_path = cache_dir / team_filename

                with open(team_path, 'w', encoding='utf-8') as f:
                    # Write team in simple format (no header)
                    for j, pokemon in enumerate(team.pokemon):
                        if j > 0:
                            f.write("\n")

                        f.write(f"{pokemon['name']}\n")

                        # Add EVs/IVs if they exist (optional)
                        if 'evs' in pokemon and pokemon['evs']:
                            f.write(f"EVs: {pokemon['evs']}\n")
                        if 'ivs' in pokemon and pokemon['ivs']:
                            f.write(f"IVs: {pokemon['ivs']}\n")

                        # Add moves
                        for move in pokemon.get('moves', []):
                            f.write(f"- {move}\n")

            return f"""✅ Successfully exported {len(self.unique_teams)} teams to cache!

**Location:** `~/metamon_cache/teams/{team_set_name}/{battle_format}/`

**Usage in training:**
```bash
--team_set {team_set_name}
```

Teams are ready for simulated battles!"""

        except Exception as e:
            return f"Error exporting to cache: {e}"

    def get_statistics(self) -> str:
        """Get statistics about the teams"""
        if not self.unique_teams:
            return "No teams loaded"

        stats = f"## Team Statistics\n\n"
        stats += f"- **Total teams loaded:** {len(self.all_teams)}\n"
        stats += f"- **Duplicates removed:** {self.duplicate_count}\n"
        stats += f"- **Unique teams:** {len(self.unique_teams)}\n\n"

        # Count by source file
        source_counts = {}
        for team in self.unique_teams:
            source_counts[team.source_file] = source_counts.get(team.source_file, 0) + 1

        stats += "### Teams by Source File\n"
        for source, count in sorted(source_counts.items()):
            stats += f"- **{source}:** {count} teams\n"

        # Count pokemon usage
        pokemon_usage = {}
        for team in self.unique_teams:
            for pokemon in team.pokemon:
                name = pokemon['name']
                pokemon_usage[name] = pokemon_usage.get(name, 0) + 1

        stats += "\n### Top 10 Most Used Pokemon\n"
        top_pokemon = sorted(pokemon_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        for pokemon, count in top_pokemon:
            stats += f"- **{pokemon}:** {count} times\n"

        return stats


def create_dashboard():
    """Create and launch the Gradio dashboard"""

    manager = TeamManager()

    with gr.Blocks(title="Metamon Team Manager") as demo:
        gr.Markdown("# 🎮 Metamon Team Manager")
        gr.Markdown("Manage and curate Pokemon teams from world-class players")

        with gr.Row():
            teams_dir_input = gr.Textbox(
                value="teams/gen1ou",
                label="Teams Directory",
                placeholder="teams/gen1ou"
            )
            load_btn = gr.Button("🔄 Load Teams", variant="primary")

        load_status = gr.Markdown()

        with gr.Row():
            remove_dupes_btn = gr.Button("🧹 Remove Duplicates", variant="secondary")

        dedup_status = gr.Markdown()

        with gr.Row():
            expected_pokemon_input = gr.Number(
                label="Expected Pokémon per Team",
                value=6,
                precision=0,
                minimum=1,
                maximum=6
            )
            expected_moves_input = gr.Number(
                label="Expected Moves per Pokémon",
                value=4,
                precision=0,
                minimum=1,
                maximum=4
            )
            validate_btn = gr.Button("✅ Validate & Filter Teams", variant="secondary")

        validation_status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("📊 Team List"):
                refresh_table_btn = gr.Button("🔄 Refresh Table")
                teams_table = gr.Dataframe(
                    headers=["ID", "Name", "Format", "Pokemon (preview)", "Count", "Source"],
                    datatype=["number", "str", "str", "str", "number", "str"],
                    label="Unique Teams"
                )

            with gr.Tab("🔍 Team Details"):
                with gr.Row():
                    team_idx_input = gr.Number(
                        label="Team ID",
                        value=0,
                        precision=0
                    )
                    view_team_btn = gr.Button("👁️ View Team")

                team_details = gr.Markdown()

            with gr.Tab("📈 Statistics"):
                refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
                stats_output = gr.Markdown()

            with gr.Tab("💾 Export"):
                gr.Markdown("### Export to Metamon Cache (For Training)")
                gr.Markdown("Export teams in the proper format for use with `--team_set` argument")

                with gr.Row():
                    team_set_name_input = gr.Textbox(
                        value="curated_gen1ou",
                        label="Team Set Name",
                        placeholder="e.g., curated_gen1ou, pro_teams, custom_meta"
                    )
                    battle_format_input = gr.Textbox(
                        value="gen1ou",
                        label="Battle Format",
                        placeholder="gen1ou"
                    )

                export_cache_btn = gr.Button("🚀 Export to Cache (~/metamon_cache/teams)", variant="primary")
                cache_export_status = gr.Markdown()

                gr.Markdown("---")
                gr.Markdown("### Export as Files (For Backup)")

                with gr.Row():
                    export_txt_input = gr.Textbox(
                        value="teams/gen1ou/curated_teams.txt",
                        label="Export to .txt (Showdown format)"
                    )
                    export_txt_btn = gr.Button("📄 Export TXT")

                with gr.Row():
                    export_json_input = gr.Textbox(
                        value="teams/gen1ou/curated_teams.json",
                        label="Export to JSON"
                    )
                    export_json_btn = gr.Button("📋 Export JSON")

                export_status = gr.Markdown()

        # Event handlers
        def load_teams_handler(teams_dir):
            manager.teams_dir = Path(teams_dir)
            result = manager.load_teams()
            return result

        def remove_duplicates_handler():
            return manager.remove_duplicates()

        def validate_teams_handler(expected_pokemon, expected_moves):
            return manager.apply_validation(int(expected_pokemon), int(expected_moves))

        def refresh_table_handler():
            return manager.get_teams_summary()

        def view_team_handler(team_idx):
            return manager.get_team_details(int(team_idx))

        def refresh_stats_handler():
            return manager.get_statistics()

        def export_txt_handler(output_file):
            return manager.export_teams(output_file)

        def export_json_handler(output_file):
            return manager.export_json(output_file)

        def export_cache_handler(team_set_name, battle_format):
            return manager.export_to_cache(team_set_name, battle_format)

        # Connect events
        load_btn.click(
            fn=load_teams_handler,
            inputs=[teams_dir_input],
            outputs=[load_status]
        )

        remove_dupes_btn.click(
            fn=remove_duplicates_handler,
            outputs=[dedup_status]
        )

        validate_btn.click(
            fn=validate_teams_handler,
            inputs=[expected_pokemon_input, expected_moves_input],
            outputs=[validation_status]
        )

        refresh_table_btn.click(
            fn=refresh_table_handler,
            outputs=[teams_table]
        )

        view_team_btn.click(
            fn=view_team_handler,
            inputs=[team_idx_input],
            outputs=[team_details]
        )

        refresh_stats_btn.click(
            fn=refresh_stats_handler,
            outputs=[stats_output]
        )

        export_txt_btn.click(
            fn=export_txt_handler,
            inputs=[export_txt_input],
            outputs=[export_status]
        )

        export_json_btn.click(
            fn=export_json_handler,
            inputs=[export_json_input],
            outputs=[export_status]
        )

        export_cache_btn.click(
            fn=export_cache_handler,
            inputs=[team_set_name_input, battle_format_input],
            outputs=[cache_export_status]
        )

    return demo


if __name__ == "__main__":
    demo = create_dashboard()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
