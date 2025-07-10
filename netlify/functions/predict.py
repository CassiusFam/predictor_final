import json
import sys
import os
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import requests
from cachetools import TTLCache, cached

# Add the project root to Python path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from utils.airport import code_to_name, get_all_airport_details, code_to_latlon
except ImportError:
    # Fallback implementations
    def code_to_name(code):
        return f"Airport {code}"
    
    def get_all_airport_details(code):
        return {
            'iata': code,
            'name': f"Airport {code}",
            'city': f"City {code}",
            'country': f"Country {code}"
        }
    
    def code_to_latlon(code):
        return (0.0, 0.0)

# Constants for delay categories
REGULAR_DELAY = 15    # minutes
SEVERE_DELAY = 60     # minutes
EXTREME_DELAY = 180   # minutes

# Cache for model and encoder
_model = None
_encoder = None

def load_model_and_encoder():
    """Load the trained model and encoder"""
    global _model, _encoder
    
    if _model is not None and _encoder is not None:
        return _model, _encoder
    
    # Try lightweight models first (for deployment)
    lite_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_model_lite.pkl')
    lite_encoder_path = os.path.join(os.path.dirname(__file__), '..', '..', 'airport_encoder_lite.pkl')
    
    if os.path.exists(lite_model_path) and os.path.exists(lite_encoder_path):
        with open(lite_model_path, 'rb') as f:
            _model = pickle.load(f)
        with open(lite_encoder_path, 'rb') as f:
            _encoder = pickle.load(f)
        return _model, _encoder
    
    # Fallback to full models
    model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'trained_model.pkl')
    encoder_path = os.path.join(os.path.dirname(__file__), '..', '..', 'airport_encoder.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        with open(model_path, 'rb') as f:
            _model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            _encoder = pickle.load(f)
        return _model, _encoder
    
    raise FileNotFoundError("No trained model found")

# Weather cache (1 hour TTL)
weather_cache = TTLCache(maxsize=1000, ttl=3600)

@cached(weather_cache)
def get_weather_info(lat, lon):
    """Get weather information for given coordinates"""
    try:
        # Use OpenWeatherMap API (you'll need to set the API key)
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if not api_key:
            return None, True, "No API Key"
        
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                'condition': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'temperature': data['main']['temp'] - 273.15,  # Convert from Kelvin to Celsius
                'humidity': data['main']['humidity'],
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'wind_speed': data['wind']['speed']
            }
            
            # Determine if weather might cause delays
            condition = data['weather'][0]['main'].lower()
            has_delay = condition in ['rain', 'snow', 'thunderstorm', 'mist', 'fog']
            
            return weather_info, False, "OpenWeatherMap"
        
    except Exception as e:
        logging.error(f"Weather API error: {e}")
    
    # Return default weather data
    return {
        'condition': 'Clear',
        'description': 'clear sky',
        'temperature': 20.0,
        'humidity': 50,
        'visibility': 10.0,
        'wind_speed': 5.0
    }, True, "Default"

def predict_delay(origin, destination, departure_time):
    """Make delay prediction"""
    try:
        model, encoder = load_model_and_encoder()
        
        # Parse departure time
        if isinstance(departure_time, str):
            departure_dt = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
        else:
            departure_dt = departure_time
        
        hour = departure_dt.hour
        
        # Encode airports
        try:
            origin_encoded = encoder.transform([origin])[0]
            dest_encoded = encoder.transform([destination])[0]
        except ValueError as e:
            return {
                'error': f"Unknown airport code: {str(e)}",
                'status': 400
            }
        
        # Get weather information
        origin_lat, origin_lon = code_to_latlon(origin)
        weather_info, used_default, provider_name = get_weather_info(origin_lat, origin_lon)
        
        # Determine weather delay flag
        has_weather_delay = 0
        if not used_default and weather_info:
            condition = weather_info.get('condition', '').lower()
            has_weather_delay = 1 if condition in ['rain', 'snow', 'thunderstorm', 'mist', 'fog'] else 0
        
        # Make prediction
        X_pred_df = pd.DataFrame(
            [[origin_encoded, dest_encoded, hour, has_weather_delay]],
            columns=['Origin_encoded', 'Dest_encoded', 'Hour', 'Has_Weather_Delay']
        )
        
        prediction_probs = model.predict_proba(X_pred_df)[0]
        model_categories = model.classes_
        
        # Create ordered predictions
        DESIRED_CATEGORY_ORDER = ['No Delay', 'Regular Delay', 'Severe Delay', 'Extreme Delay']
        prob_map = {cat: prob for cat, prob in zip(model_categories, prediction_probs)}
        
        predictions_ordered = []
        for cat_name in DESIRED_CATEGORY_ORDER:
            predictions_ordered.append({
                'category': cat_name,
                'probability': prob_map.get(cat_name, 0.0)
            })
        
        # Get airport names
        origin_details = get_all_airport_details(origin)
        dest_details = get_all_airport_details(destination)
        
        return {
            'predictions': predictions_ordered,
            'weather': weather_info,
            'used_default_weather': used_default,
            'weather_provider': provider_name,
            'origin_iata': origin,
            'origin_name': origin_details.get('name', f"Airport {origin}"),
            'destination_iata': destination,
            'destination_name': dest_details.get('name', f"Airport {destination}")
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {
            'error': str(e),
            'status': 500
        }

def handler(event, context):
    """Netlify function handler"""
    try:
        # Handle CORS preflight
        if event['httpMethod'] == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                },
                'body': ''
            }
        
        if event['httpMethod'] != 'POST':
            return {
                'statusCode': 405,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Method not allowed'})
            }
        
        # Parse request body
        body = json.loads(event['body'])
        origin = body.get('origin')
        destination = body.get('destination')
        departure_time = body.get('departure_time')
        
        if not all([origin, destination, departure_time]):
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Missing required fields'})
            }
        
        # Make prediction
        result = predict_delay(origin, destination, departure_time)
        
        if 'error' in result:
            status_code = result.get('status', 500)
            return {
                'statusCode': status_code,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': result['error']})
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }
