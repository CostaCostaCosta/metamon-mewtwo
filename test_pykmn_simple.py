"""
Extremely simple test to find where pypkmn crashes.
"""

from pykmn.engine.gen1 import Battle, Player, Pokemon
from pykmn.engine.common import ResultType

print("Creating team...")
team = [
    Pokemon(species="Tauros", moves=("Body Slam", "Hyper Beam", "Blizzard", "Earthquake")),
    Pokemon(species="Snorlax", moves=("Body Slam", "Earthquake", "Rest", "Ice Beam")),
    Pokemon(species="Chansey", moves=("Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled")),
    Pokemon(species="Exeggutor", moves=("Psychic", "Sleep Powder", "Explosion", "Stun Spore")),
    Pokemon(species="Starmie", moves=("Thunderbolt", "Blizzard", "Thunder Wave", "Recover")),
    Pokemon(species="Alakazam", moves=("Psychic", "Seismic Toss", "Thunder Wave", "Recover")),
]
print(f"Team created: {len(team)} Pokemon")

print("\nCreating battle...")
battle = Battle(team, team)
print("Battle created")

print("\nInitializing with team preview...")
result, _ = battle.update_raw(0, 0)
print(f"Initialized, result type: {result.type()}")

print("\nGetting legal choices...")
try:
    legal_p1 = battle.possible_choices_raw(Player.P1, result)
    print(f"P1 legal choices: {legal_p1}")
except Exception as e:
    print(f"ERROR getting P1 choices: {e}")
    raise

try:
    legal_p2 = battle.possible_choices_raw(Player.P2, result)
    print(f"P2 legal choices: {legal_p2}")
except Exception as e:
    print(f"ERROR getting P2 choices: {e}")
    raise

print("\nTaking first turn...")
choice_p1 = legal_p1[0]
choice_p2 = legal_p2[0]
print(f"P1 choice: {choice_p1}, P2 choice: {choice_p2}")

try:
    result, _ = battle.update_raw(choice_p1, choice_p2)
    print(f"Turn completed, result type: {result.type()}")
except Exception as e:
    print(f"ERROR on update: {e}")
    raise

print("\nSuccess! Basic pypkmn operations work.")
