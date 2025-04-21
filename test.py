import pandas as pd
import json

# Define file path
file_path = "possitive_affirmation.csv"  # Ensure correct filename

# Load the CSV file
df = pd.read_csv(file_path)

# Convert dataframe to JSON format
json_data = df.to_json(orient="records", indent=4)

# Save JSON to a file
json_file_path = "affirmations.json"
with open(json_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_data)

print(f"✅ JSON file created: {json_file_path}")
