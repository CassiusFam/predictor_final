import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import pickle
import logging

# Constants for delay categories
REGULAR_DELAY = 15    # minutes
SEVERE_DELAY = 60     # minutes
EXTREME_DELAY = 180   # minutes

def load_data():
    """Load and return the flight data (final_data.csv)"""
    return pd.read_csv('final_data.csv')

def train_and_save_model():
    """Train the model and save it to disk, return model and encoder"""
    # Load and preprocess data
    df = load_data()
    
    # Preprocess data
    df['Delay_Minutes'] = df['DEP_DELAY']
    df['Hour'] = df['HOUR_BIN']
    df = df.dropna(subset=['DEP_DELAY', 'ORIGIN', 'DEST', 'HOUR_BIN'])
    
    # Create encoder for airport codes
    all_airports = sorted(list(pd.concat([df['ORIGIN'], df['DEST']]).unique()))
    airport_encoder_instance = LabelEncoder()
    airport_encoder_instance.fit(all_airports)

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
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Calibrate the Random Forest model
    calibrated_rf_model = CalibratedClassifierCV(
        rf_model, 
        method='isotonic',
        cv=3,
        n_jobs=-1
    )
    calibrated_rf_model.fit(X_calib, y_calib)

    # Evaluate calibration
    y_calib_pred_probs = calibrated_rf_model.predict_proba(X_calib)
    
    brier = brier_score_loss(y_calib, y_calib_pred_probs)
    logloss = log_loss(y_calib, y_calib_pred_probs)
    
    logging.info(f"Calibrated Model Brier Score (on calibration set): {brier:.4f}")
    logging.info(f"Calibrated Model Log-Loss (on calibration set): {logloss:.4f}")

    # Save the trained model and encoder
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(calibrated_rf_model, f)
    
    with open('airport_encoder.pkl', 'wb') as f:
        pickle.dump(airport_encoder_instance, f)
    
    print("Model and encoder saved successfully!")
    
    return calibrated_rf_model, airport_encoder_instance
