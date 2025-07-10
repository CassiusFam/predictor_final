import pandas as pd
import json
import pathlib

# Define paths relative to this script or use absolute paths
PROJECT_ROOT = pathlib.Path(__file__).parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "airports_raw.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "data" / "airport_codes.json"

def transform_data():
    print(f"Reading raw CSV from: {RAW_CSV_PATH}")
    try:
        df = pd.read_csv(RAW_CSV_PATH, usecols=["iata_code", "name", "latitude_deg", "longitude_deg"])
    except ValueError as e:
        print(f"Error reading CSV. It's possible some columns are missing or the file is empty: {e}")
        print("Please ensure 'iata_code', 'name', 'latitude_deg', 'longitude_deg' are in the CSV.")
        # Attempt to read with only essential columns if specific error occurs
        if "usecols do not match columns" in str(e):
            try:
                print("Attempting to read only 'iata_code' and 'name' due to previous error.")
                df = pd.read_csv(RAW_CSV_PATH, usecols=["iata_code", "name"])
                # Add missing lat/lon columns with NaNs if they were the issue
                if "latitude_deg" not in df.columns:
                    df["latitude_deg"] = pd.NA
                if "longitude_deg" not in df.columns:
                    df["longitude_deg"] = pd.NA
            except Exception as e2:
                print(f"Failed to read even minimal columns: {e2}")
                return
        else:
            return

    df = df.dropna(subset=["iata_code", "name"])
    df.columns = ["IATA", "name", "lat", "lon"]
    
    # Ensure IATA codes are uppercase and strings to prevent issues with mixed types or numbers
    df["IATA"] = df["IATA"].astype(str).str.upper()
    
    # Handle potential duplicate IATA codes: keep the first occurrence
    df = df.drop_duplicates(subset=["IATA"], keep="first")

    mapping = df.set_index("IATA").to_dict(orient="index")
    
    JSON_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure data directory exists
    JSON_OUTPUT_PATH.write_text(json.dumps(mapping, indent=2))
    print(f"Successfully created skinny JSON: {JSON_OUTPUT_PATH}")
    print(f"Total airports processed: {len(mapping)}")

if __name__ == "__main__":
    transform_data()
