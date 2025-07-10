#!/usr/bin/env python3
"""
Standalone script to train and save the flight delay prediction model.
Run this once to generate the trained_model.pkl and airport_encoder.pkl files.

Usage: python train_model_standalone.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for delay categories
REGULAR_DELAY = 15    # minutes
SEVERE_DELAY = 60     # minutes
EXTREME_DELAY = 180   # minutes

def load_data():
    """Load and return the flight data (final_data.csv)"""
    return pd.read_csv('final_data.csv')

def train_and_save_model():
    """Train the model and save it to disk, return model and encoder"""
    logging.info("Loading flight data...")
    df = load_data()
    
    # Preprocess data
    df['Delay_Minutes'] = df['DEP_DELAY']
    df['Hour'] = df['HOUR_BIN']
    df = df.dropna(subset=['DEP_DELAY', 'ORIGIN', 'DEST', 'HOUR_BIN'])
    
    logging.info(f"Data loaded and preprocessed. Shape: {df.shape}")
    
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
    
    logging.info("Data preprocessing completed.")
    logging.info(f"Delay category distribution:\n{df['Delay_Category'].value_counts()}")
    
    # Train model
    features = ['Origin_encoded', 'Dest_encoded', 'Hour', 'Has_Weather_Delay']
    X = df[features]
    y = df['Delay_Category']

    # Split data into training and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Training Random Forest model...")
    # Train Random Forest with class_weight='balanced'
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42, 
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    logging.info("Calibrating model...")
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
    logging.info("Saving trained model...")
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(calibrated_rf_model, f)
    
    with open('airport_encoder.pkl', 'wb') as f:
        pickle.dump(airport_encoder_instance, f)
    
    print("✅ Model and encoder saved successfully!")
    print("📁 Files created:")
    print("   - trained_model.pkl")
    print("   - airport_encoder.pkl")
    print("🚀 You can now run the Flask app without needing to load the large CSV file!")
    
    return calibrated_rf_model, airport_encoder_instance

if __name__ == "__main__":
    print("🔄 Training flight delay prediction model...")
    print("⏳ This may take a few minutes...")
    try:
        train_and_save_model()
    except FileNotFoundError:
        print("❌ Error: final_data.csv not found!")
        print("   Make sure the data file is in the current directory.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        logging.error(f"Training failed: {e}")
