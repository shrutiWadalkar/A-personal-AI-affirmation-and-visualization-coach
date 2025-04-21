import json

# Load JSON data from file
with open("affirmations.json", "r", encoding="utf-8") as file:
    affirmations_data = json.load(file)  # Load JSON into a Python object

# Print the number of entries
print(f"Affirmations Data Loaded: {len(affirmations_data)} entries")

# Print a few entries to verify
print(affirmations_data[:5])  # Print first 5 affirmations
