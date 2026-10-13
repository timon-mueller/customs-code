from pathlib import Path

# Target directory
target_dir = Path("dataset/example_data")

# Create empty JSON files from 04.json to 100.json
for i in range(4, 101):
    file_path = target_dir / f"{i:02d}.json"

    # Create only if it doesn't already exist
    if not file_path.exists():
        file_path.write_text("{}", encoding="utf-8")

print("Created JSON files 04.json to 100.json")