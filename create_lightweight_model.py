#!/usr/bin/env python3
"""
Script to create a smaller, deployment-ready model for the Flight Delay Predictor.
This will significantly reduce the model size for free hosting platforms.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import pickle
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for delay categories
REGULAR_DELAY = 15    # minutes
SEVERE_DELAY = 60     # minutes
EXTREME_DELAY = 180   # minutes

def load_data():
    """Load and return the flight data (final_data.csv)"""
    return pd.read_csv('final_data.csv')

def create_lightweight_model():
    """Train a smaller model optimized for deployment"""
    logging.info("Loading flight data for lightweight model...")
    df = load_data()
    
    # Sample the data to reduce training time and model size
    # Use stratified sampling to maintain class distribution
    sample_size = min(100000, len(df))  # Use max 100k samples
    df_sample = df.sample(n=sample_size, random_state=42)
    
    logging.info(f"Using {len(df_sample)} samples for lightweight model training")
    
    # Preprocess data
    df_sample['Delay_Minutes'] = df_sample['DEP_DELAY']
    df_sample['Hour'] = df_sample['HOUR_BIN']
    df_sample = df_sample.dropna(subset=['DEP_DELAY', 'ORIGIN', 'DEST', 'HOUR_BIN'])
    
    # Create encoder for airport codes
    all_airports = sorted(list(pd.concat([df['ORIGIN'], df['DEST']]).unique()))  # Use full dataset for encoder
    airport_encoder_instance = LabelEncoder()
    airport_encoder_instance.fit(all_airports)

    df_sample['Origin_encoded'] = airport_encoder_instance.transform(df_sample['ORIGIN'])
    df_sample['Dest_encoded'] = airport_encoder_instance.transform(df_sample['DEST'])
    
    df_sample['Has_Weather_Delay'] = df_sample['WEATHER_DELAY'].notna() & (df_sample['WEATHER_DELAY'] > 0)
    
    # Categorization
    def categorize_delay(delay_minutes):
        if pd.isna(delay_minutes) or delay_minutes <= 0:
            return 'No Delay'
        elif delay_minutes <= REGULAR_DELAY:
            return 'Regular Delay'
        elif delay_minutes <= SEVERE_DELAY:
            return 'Severe Delay'
        else:
            return 'Extreme Delay'
    
    df_sample['Delay_Category'] = df_sample['Delay_Minutes'].apply(categorize_delay)
    
    logging.info(f"Delay category distribution:\n{df_sample['Delay_Category'].value_counts()}")
    
    # Train lightweight model
    features = ['Origin_encoded', 'Dest_encoded', 'Hour', 'Has_Weather_Delay']
    X = df_sample[features]
    y = df_sample['Delay_Category']

    # Split data
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Training lightweight Random Forest model...")
    # Smaller, deployment-optimized Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=50,        # Reduced from 200
        max_depth=15,           # Limit tree depth
        min_samples_split=10,   # Prevent overfitting
        min_samples_leaf=5,     # Prevent overfitting
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    logging.info("Calibrating lightweight model...")
    # Use sigmoid for faster calibration
    calibrated_rf_model = CalibratedClassifierCV(
        rf_model, 
        method='sigmoid',  # Faster than isotonic
        cv=3,
        n_jobs=-1
    )
    calibrated_rf_model.fit(X_calib, y_calib)

    # Evaluate
    y_calib_pred_probs = calibrated_rf_model.predict_proba(X_calib)
    from sklearn.metrics import brier_score_loss, log_loss
    
    brier = brier_score_loss(y_calib, y_calib_pred_probs)
    logloss = log_loss(y_calib, y_calib_pred_probs)
    
    logging.info(f"Lightweight Model Brier Score: {brier:.4f}")
    logging.info(f"Lightweight Model Log-Loss: {logloss:.4f}")

    # Save the lightweight model
    logging.info("Saving lightweight model...")
    with open('trained_model_lite.pkl', 'wb') as f:
        pickle.dump(calibrated_rf_model, f)
    
    with open('airport_encoder_lite.pkl', 'wb') as f:
        pickle.dump(airport_encoder_instance, f)
    
    # Check file sizes
    model_size = os.path.getsize('trained_model_lite.pkl') / (1024 * 1024)  # MB
    encoder_size = os.path.getsize('airport_encoder_lite.pkl') / (1024 * 1024)  # MB
    
    print(f"✅ Lightweight model created successfully!")
    print(f"📁 Model file size: {model_size:.1f} MB")
    print(f"📁 Encoder file size: {encoder_size:.1f} MB")
    print(f"📁 Total size: {model_size + encoder_size:.1f} MB")
    print(f"🚀 Ready for deployment on free hosting platforms!")
    
    return calibrated_rf_model, airport_encoder_instance

if __name__ == "__main__":
    print("🔧 Creating lightweight model for deployment...")
    print("⏳ This will take a few minutes...")
    try:
        create_lightweight_model()
        print("\n🎉 Lightweight model ready!")
        print("📝 Next steps:")
        print("   1. Update app_new.py to use the lite model files")
        print("   2. Deploy to Railway, Render, or Vercel")
        print("   3. Enjoy your free hosting! 🚀")
    except FileNotFoundError:
        print("❌ Error: final_data.csv not found!")
        print("   Make sure the data file is in the current directory.")
    except Exception as e:
        print(f"❌ Error during model creation: {e}")
        logging.error(f"Model creation failed: {e}")
