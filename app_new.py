from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Added
from sklearn.calibration import CalibratedClassifierCV, calibration_curve # Added
from sklearn.metrics import brier_score_loss, log_loss # Added
import requests
import datetime
import json
import logging
import pickle
import os
from cachetools import TTLCache, cached
from utils.airport import code_to_name, get_all_airport_details, code_to_latlon

app = Flask(__name__)

# Constants for delay categories
REGULAR_DELAY = 15    # minutes
SEVERE_DELAY = 60     # minutes
EXTREME_DELAY = 180   # minutes

def load_or_train_model():
    """Load pre-trained model and encoder, or train new ones if not found"""
    # Try lightweight models first (for deployment)
    lite_model_path = 'trained_model_lite.pkl'
    lite_encoder_path = 'airport_encoder_lite.pkl'
    
    if os.path.exists(lite_model_path) and os.path.exists(lite_encoder_path):
        # Load lightweight model and encoder
        logging.info("Loading lightweight model and encoder for deployment...")
        with open(lite_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(lite_encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        logging.info("Lightweight model and encoder loaded successfully!")
        return model, encoder
    
    # Fallback to full models
    model_path = 'trained_model.pkl'
    encoder_path = 'airport_encoder.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        # Load pre-trained model and encoder
        logging.info("Loading full pre-trained model and encoder...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        logging.info("Full model and encoder loaded successfully!")
        return model, encoder
    else:
        # Train new model if no files exist
        logging.info("No pre-trained model found. Training new model...")
        return train_model()

def load_data():
    """Load and return the flight data (final_data.csv)"""
    return pd.read_csv('final_data.csv')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for weather fetching
WTTR_TEMPLATES = [
    "https://wttr.in/{lat},{lon}?format=j1",        # plain lat,lon
    "https://wttr.in/~{lat},{lon}?format=j1",       # tilde lat,lon
    "https://wttr.in/{iata}?format=j1"              # airport code
]
OPEN_METEO_URL = ("https://api.open-meteo.com/v1/forecast"
                  "?latitude={lat}&longitude={lon}"
                  "&current_weather=true"
                  "&hourly=temperature_2m,precipitation,wind_speed_10m"
                  "&timezone=UTC")

def _safe_json(resp: requests.Response) -> dict:
    """Return JSON if body starts with '{', else raise ValueError."""
    text_content = resp.text.strip()
    if not text_content.startswith("{"):
        raise ValueError(f"Response body is not JSON (starts with: '{text_content[:30]}...'). Full text (first 200 chars): {text_content[:200]}")
    return json.loads(text_content) # Use json.loads for consistency with json.JSONDecodeError

def _parse_wttr(raw_data: dict) -> dict:
    current = raw_data['current_condition'][0]
    return {
        'temperature': float(current['temp_C']),
        'max_temperature': float(current['temp_C']) + 2, # Approximation, as in previous logic
        'precipitation': float(current['precipMM']),
        'wind_speed': float(current['windspeedKmph']) * 0.277778, # Kmph to m/s
        'conditions': current['weatherDesc'][0]['value']
    }

def _parse_openmeteo(raw_data: dict) -> dict:
    cw = raw_data["current_weather"]
    hourly = raw_data["hourly"]
    
    weather_code = cw.get("weathercode", -1)
    conditions_map = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }
    conditions_str = conditions_map.get(weather_code, f"Weather code {weather_code}")

    max_temp = cw["temperature"] 
    if hourly.get("temperature_2m") and len(hourly["temperature_2m"]) > 0:
        max_temp = max(hourly["temperature_2m"])
    
    precipitation_val = 0.0
    if hourly.get("precipitation") and len(hourly["precipitation"]) > 0:
        precipitation_val = max(hourly.get("precipitation", [0.0]))


    return {
        "temperature": cw["temperature"],
        "max_temperature": max_temp,
        "precipitation": precipitation_val, 
        "wind_speed": cw["windspeed"], # Assumed to be m/s as per Open-Meteo docs for wind_speed_10m
        "conditions": conditions_str
    }

def train_model():
    """Train the model and return model and encoder"""
    # Load and preprocess data
    df = load_data() # Local df for training
    
    # Preprocess data
    df['Delay_Minutes'] = df['DEP_DELAY']
    df['Hour'] = df['HOUR_BIN']
    df = df.dropna(subset=['DEP_DELAY', 'ORIGIN', 'DEST', 'HOUR_BIN'])
    
    # Create encoder for airport codes
    all_airports = sorted(list(pd.concat([df['ORIGIN'], df['DEST']]).unique()))
    airport_encoder_instance = LabelEncoder()
    airport_encoder_instance.fit(all_airports) # Fit on ALL unique airports

    df['Origin_encoded'] = airport_encoder_instance.transform(df['ORIGIN'])
    df['Dest_encoded'] = airport_encoder_instance.transform(df['DEST'])
    
    df['Has_Weather_Delay'] = df['WEATHER_DELAY'].notna() & (df['WEATHER_DELAY'] > 0)
    
    # Updated categorization to include "No Delay" 
    def categorize_delay(delay_minutes):
        if pd.isna(delay_minutes) or delay_minutes <= 0:
            return 'No Delay'
        elif delay_minutes <= REGULAR_DELAY:
            return 'Regular Delay'
        elif delay_minutes <= SEVERE_DELAY:
            return 'Severe Delay'
        else:
            return 'Extreme Delay'
    
    df['Delay_Category'] = df['Delay_Minutes'].apply(categorize_delay)
    
    # Train model
    features = ['Origin_encoded', 'Dest_encoded', 'Hour', 'Has_Weather_Delay']
    X = df[features]
    y = df['Delay_Category']

    # Split data into training and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest with class_weight='balanced'
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced' # Added
    )
    rf_model.fit(X_train, y_train)
    
    # Calibrate the Random Forest model
    # Use the rf_model trained on X_train, and calibrate it using X_calib with 3-fold CV for the calibrator
    calibrated_rf_model = CalibratedClassifierCV(
        rf_model, 
        method='isotonic', # Changed from 'sigmoid' to 'isotonic'
        cv=3, # 3-fold CV on X_calib to train the calibrator
        n_jobs=-1
    )
    calibrated_rf_model.fit(X_calib, y_calib)

    # Evaluate calibration (optional, but good for logging)
    y_calib_pred_probs = calibrated_rf_model.predict_proba(X_calib)
    
    brier = brier_score_loss(y_calib, y_calib_pred_probs) # Multi-class Brier score
    logloss = log_loss(y_calib, y_calib_pred_probs)
    
    logging.info(f"Calibrated Model Brier Score (on calibration set): {brier:.4f}")
    logging.info(f"Calibrated Model Log-Loss (on calibration set): {logloss:.4f}")

    logging.info("Reliability Diagram Data (prob_true, prob_pred) per class (3 bins):")
    for i, class_name in enumerate(calibrated_rf_model.classes_):
        y_true_class = (y_calib == class_name).astype(int)
        y_prob_class = y_calib_pred_probs[:, i]
        # Handle cases where a class might not be present in y_calib for a bin, or y_prob_class has no variance for a bin
        try:
            prob_true, prob_pred = calibration_curve(y_true_class, y_prob_class, n_bins=3, strategy='uniform')
            logging.info(f"  Class '{class_name}': prob_true={np.round(prob_true, 4)}, prob_pred={np.round(prob_pred, 4)}")
        except ValueError as e:
            logging.warning(f"Could not compute calibration curve for class '{class_name}': {e}")
            logging.warning(f"    Unique y_true_class values: {np.unique(y_true_class, return_counts=True)}")
            logging.warning(f"    Unique y_prob_class values (approx first 10): {np.unique(np.round(y_prob_class[:10],3))}")


    # The prompt mentioned isotonic as an alternative if Brier > 0.20.
    # This can be implemented as a next step if sigmoid isn't sufficient.
    if brier > 0.20:
        logging.warning(f"Brier score {brier:.4f} is > 0.20. Isotonic calibration was attempted. Further review might be needed if score remains high.") # Updated warning

    # Save the trained model and encoder
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(calibrated_rf_model, f)
    
    with open('airport_encoder.pkl', 'wb') as f:
        pickle.dump(airport_encoder_instance, f)
    
    logging.info("Model and encoder saved successfully!")

    return calibrated_rf_model, airport_encoder_instance

# Load or train model and encoder
model, airport_encoder = load_or_train_model()

# Load airport data through the centralized system
airport_details_for_template = get_all_airport_details()
logging.info(f"Successfully loaded {len(airport_details_for_template)} airport details for UI dropdowns.")


@cached(cache=TTLCache(maxsize=500, ttl=3600)) # Cache for 1 hour
def get_weather_data(airport_code: str) -> tuple[dict, bool, str]:
    """
    Fetch weather data, trying wttr.in, then Open-Meteo, then defaults.
    Uses code_to_latlon from utils.airport for coordinates.
    Returns: (weather_dict, used_default_flag, provider_name_str)
    """
    lat, lon = None, None
    lat_s, lon_s = None, None

    # Get coordinates from the new centralized system
    lat, lon = code_to_latlon(airport_code.strip())
    if lat is not None and lon is not None:
        lat_s, lon_s = f"{lat:.4f}", f"{lon:.4f}"
        logging.info(f"Coordinates for {airport_code} from utils.airport: Lat={lat_s}, Lon={lon_s}")
    else:
        logging.warning(f"Airport code '{airport_code}' not found or missing coordinates in utils.airport. Weather fetching will rely on IATA lookup or defaults.")

    # 1) Try wttr.in variants
    # Construct URLs to try: lat/lon, ~lat/lon (if lat/lon available), then IATA code
    urls_for_wttr = []
    provider_details_wttr = []


    if lat_s and lon_s:
        urls_for_wttr.append(WTTR_TEMPLATES[0].format(lat=lat_s, lon=lon_s, iata=airport_code))
        provider_details_wttr.append(f"wttr.in (lat,lon: {lat_s},{lon_s})")
        
        urls_for_wttr.append(WTTR_TEMPLATES[1].format(lat=lat_s, lon=lon_s, iata=airport_code))
        provider_details_wttr.append(f"wttr.in (~lat,lon: {lat_s},{lon_s})")

    urls_for_wttr.append(WTTR_TEMPLATES[2].format(iata=airport_code, lat=lat_s or "0", lon=lon_s or "0")) # lat/lon not strictly needed by this template string
    provider_details_wttr.append(f"wttr.in (IATA: {airport_code})")
    
    for i, url in enumerate(urls_for_wttr):
        try:
            logging.info(f"Attempting wttr.in: {url}")
            r = requests.get(url, timeout=(3, 6)) # (connect_timeout, read_timeout)
            r.raise_for_status() # Check for HTTP errors like 4xx, 5xx
            data = _safe_json(r) # Raises ValueError if not JSON
            weather_info = _parse_wttr(data)
            return weather_info, False, provider_details_wttr[i]
        except (requests.RequestException, ValueError, json.JSONDecodeError, KeyError) as e:
            # KeyError for parsing issues within _parse_wttr if structure is unexpected
            logging.warning(f"wttr.in failed for {url} ({provider_details_wttr[i]}): {e}")
            if isinstance(e, ValueError) and "Body is not JSON" in str(e):
                logging.warning(f"Response text from wttr.in that was not JSON: {r.text[:200]}") 


    # 2) Fallback to Open-Meteo (only if lat/lon are available)
    if lat_s and lon_s:
        open_meteo_req_url = OPEN_METEO_URL.format(lat=lat_s, lon=lon_s)
        try:
            logging.info(f"Attempting Open-Meteo: {open_meteo_req_url}")
            r = requests.get(open_meteo_req_url, timeout=(3, 6))
            r.raise_for_status()
            data = r.json() # Open-Meteo is expected to always return JSON if status is OK
            weather_info = _parse_openmeteo(data)
            return weather_info, False, f"Open-Meteo ({lat_s},{lon_s})"
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Open-Meteo failed for {lat_s},{lon_s}: {e}")
    else:
        logging.warning(f"Skipping Open-Meteo for {airport_code} due to missing coordinates.")

    # 3) Ultimate fallback – defaults
    logging.warning(f"All weather services failed for {airport_code}. Using default weather.")
    return {
        'temperature': 15.0, 'max_temperature': 20.0, 'precipitation': 0.0,
        'wind_speed': 3.0, 'conditions': 'Clear'
    }, True, "Default (all services unavailable)"

@app.route('/')
def home():
    # Use the airport list derived from the centralized system
    return render_template('index.html', airports=airport_details_for_template)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {'status': 'healthy', 'model_loaded': model is not None, 'encoder_loaded': airport_encoder is not None}

@app.route('/predict', methods=['POST'])
def predict():
    origin = request.form['origin']
    destination = request.form['destination']
    hour = int(request.form['hour'])

    origin_full_name = code_to_name(origin)
    destination_full_name = code_to_name(destination)

    print(f"\\n--- New Prediction Request ---") 
    logging.info(f"Input: Origin={origin} ({origin_full_name}), Destination={destination} ({destination_full_name}), Hour={hour}")
    
    # Get weather data (this function now uses utils.airport for coords)
    weather_info, used_default, provider_name = get_weather_data(origin) 
    logging.info(f"Weather Data for {origin} (from {provider_name}): {weather_info}")
      # Prepare features
    try:
        origin_encoded = airport_encoder.transform([origin])[0]
        dest_encoded = airport_encoder.transform([destination])[0]
    except ValueError as e:
        print(f"Error encoding airports: {e}. Origin: {origin}, Dest: {destination}")
        # Handle unknown airport - perhaps return an error message to user
        # For now, this might crash if an airport not in the encoder is submitted
        # (though UI should prevent this if airport dropdowns are correct)
        return jsonify({'error': f'Invalid airport code. {e}'}), 400

    print(f"Encoded: Origin={origin_encoded}, Destination={dest_encoded}")
    
    # Weather delay flag
    has_weather_delay = False
    weather_delay_reason = "No significant weather."

    if used_default:
        has_weather_delay = False
        weather_delay_reason = f"Default weather data used (provider: {provider_name})."
    elif weather_info: # Check if weather_info is not None (it should always be a dict from get_weather_data)
        live_weather_causes_delay = False
        delay_reasons_list = []

        conditions_text = weather_info.get('conditions', '').lower()
        
        current_adverse_keywords = ['storm', 'snow', 'rain', 'fog', 'thunderstorm', 'sleet', 'hail', 
                                    'blizzard', 'hurricane', 'tornado', 'freezing rain', 
                                    'heavy snow', 'heavy rain', 'dense fog']

        for keyword in current_adverse_keywords:
            if keyword in conditions_text:
                delay_reasons_list.append(f"Condition: {weather_info.get('conditions', '')}") # Use original casing for reason
                live_weather_causes_delay = True
                break

        precip_val = weather_info.get('precipitation', 0)
        wind_val = weather_info.get('wind_speed', 0)
        temp_val = weather_info.get('temperature', 15)

        if precip_val > 5: # Threshold for precipitation in mm/hr
            delay_reasons_list.append(f"Precipitation: {precip_val:.1f}mm/hr")
            live_weather_causes_delay = True
        if wind_val > 15: # Increased threshold for wind speed in m/s (was 10)
            delay_reasons_list.append(f"Wind: {wind_val:.1f}m/s")
            live_weather_causes_delay = True
        if temp_val < 0: # Threshold for temperature in Celsius
            delay_reasons_list.append(f"Temperature: {temp_val:.1f}°C")
            live_weather_causes_delay = True
        
        has_weather_delay = live_weather_causes_delay

        if has_weather_delay and delay_reasons_list:
            weather_delay_reason = "; ".join(sorted(list(set(delay_reasons_list))))
        elif has_weather_delay:
            weather_delay_reason = "Adverse weather indicated by unspecified factors."
        else:
            weather_delay_reason = "No significant weather factors identified from live data."
    else: # Should ideally not be reached
        has_weather_delay = False
        weather_delay_reason = "Weather data was unexpectedly missing or failed to process."

    logging.info(f"Weather Delay Flag: {has_weather_delay}, Reason: {weather_delay_reason}, Provider: {provider_name}")
    
    # Make prediction
    # X_pred = np.array([[origin_encoded, dest_encoded, hour, has_weather_delay]]) # Old way
    # Create a DataFrame for prediction with feature names
    X_pred_df = pd.DataFrame(
        [[origin_encoded, dest_encoded, hour, has_weather_delay]],
        columns=['Origin_encoded', 'Dest_encoded', 'Hour', 'Has_Weather_Delay']
    )
    logging.info(f"Features for model (DataFrame):\\n{X_pred_df}") # Log the DataFrame
    
    prediction_probs = model.predict_proba(X_pred_df)[0] # Use DataFrame for prediction
    
    # Get the class labels to match with probabilities
    model_categories = model.classes_
    
    logging.info(f"Model classes (order from model.predict_proba): {model_categories}")
    logging.info(f"Raw prediction probabilities from model: {prediction_probs}")

    # Define the desired display order
    DESIRED_CATEGORY_ORDER = ['No Delay', 'Regular Delay', 'Severe Delay', 'Extreme Delay']
    
    # Create a dictionary to map category to probability from the model's output
    prob_map = {cat: prob for cat, prob in zip(model_categories, prediction_probs)}
      # Build the predictions list in the desired order
    predictions_ordered = []
    for cat_name in DESIRED_CATEGORY_ORDER:
        predictions_ordered.append({
            'category': cat_name,
            'probability': prob_map.get(cat_name, 0.0) # Default to 0.0 if a category from model.classes_ was somehow not in DESIRED_CATEGORY_ORDER (should not happen with current setup)
        })
    
    logging.info(f"Final ordered predictions for response: {predictions_ordered}")

    return jsonify({
        'predictions': predictions_ordered,
        'weather': weather_info,
        'used_default_weather': used_default,
        'weather_provider': provider_name,
        'origin_iata': origin,
        'origin_name': origin_full_name,
        'destination_iata': destination,
        'destination_name': destination_full_name
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
