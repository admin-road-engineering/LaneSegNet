import json
from pathlib import Path
import sys

def analyze_json_file(json_path):
    """Analyze a single JSON file and return its lane properties."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

    if "lanes" not in data:
        print(f"No 'lanes' key in {json_path}")
        return None

    properties_found = set()
    for lane in data["lanes"]:
        # Get all properties for this lane
        props = (
            lane.get("single", None),
            lane.get("white", None),
            lane.get("solid", None)
        )
        properties_found.add(props)
        
        # Print the first lane's vertices for verification
        if "vertices" in lane:
            print(f"\nSample vertices from {json_path}:")
            print(f"Number of vertices: {len(lane['vertices'])}")
            print(f"First few vertices: {lane['vertices'][:3]}")
            break

    return properties_found

def main():
    data_dir = Path("data/json")
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist!")
        sys.exit(1)

    # Get first few JSON files
    json_files = list(data_dir.glob("*.json"))[:5]  # Analyze first 5 files
    
    all_properties = set()
    
    print("Analyzing JSON files...")
    for json_path in json_files:
        print(f"\nAnalyzing {json_path.name}:")
        properties = analyze_json_file(json_path)
        if properties:
            all_properties.update(properties)
    
    print("\nAll unique property combinations found:")
    for props in all_properties:
        print(f"single={props[0]}, white={props[1]}, solid={props[2]}")

if __name__ == "__main__":
    main() 