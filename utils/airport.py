import json
import pathlib
import functools
import pandas as pd
import chardet

_RAW = pathlib.Path(__file__).parents[1] / "data" / "airports_raw.csv"
_JSON = _RAW.with_name("airport_codes.json")

def _convert_once():
    """Convert airports_raw.csv to UTF-8 JSON once, with automatic encoding detection."""
    if _JSON.exists():  # already cleaned
        return
    
    if not _RAW.exists():
        raise FileNotFoundError(f"Raw airport data file not found: {_RAW}")
    
    raw = _RAW.read_bytes()  # binary
    enc = chardet.detect(raw)["encoding"] or "latin1"  # guess best fit
    print(f"Detected encoding for {_RAW.name}: {enc}")
    
    df = (pd.read_csv(_RAW, encoding=enc, low_memory=False)  # one read only
          .loc[:, ["iata_code", "name", "latitude_deg", "longitude_deg"]]
          .dropna(subset=["iata_code"]))
    
    # Clean up IATA codes and ensure they're strings
    df["iata_code"] = df["iata_code"].astype(str).str.strip().str.upper()
    df = df[df["iata_code"] != ""]  # Remove empty IATA codes
    
    df.columns = ["IATA", "name", "lat", "lon"]
    
    # Convert to JSON with IATA as index
    airport_dict = df.set_index("IATA").to_dict(orient="index")
    
    _JSON.write_text(json.dumps(airport_dict, indent=2), encoding="utf-8")
    print(f"Converted {len(airport_dict)} airports to {_JSON}")

@functools.lru_cache(maxsize=1)
def _load():
    """Load airport data, converting from CSV if needed."""
    _convert_once()
    return json.loads(_JSON.read_text(encoding="utf-8"))

def code_to_name(code: str) -> str:
    """Get airport name by IATA code."""
    if not code:
        return f"Unknown ({code})"
    return _load().get(code.upper(), {}).get("name", f"Unknown ({code})")

def code_to_latlon(code: str):
    """Get (latitude, longitude) tuple by IATA code."""
    if not code:
        return (None, None)
    rec = _load().get(code.upper(), {})
    if rec:
        return (rec.get("lat"), rec.get("lon"))
    return (None, None)

def get_all_airport_details():
    """Get list of all airports with IATA and name for UI dropdowns."""
    data = _load()
    # Sort alphabetically by IATA code
    return sorted([{"iata": iata, "name": info["name"]} for iata, info in data.items()], 
                  key=lambda x: x["iata"])

