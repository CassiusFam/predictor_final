# Flight Delay Predictor ✈️

A machine learning web application that predicts flight delays using historical data and real-time weather information. Built by Cassius Famolaro as part of a machine learning course project.

## 🚀 Live Demo

[Visit the live application](your-deployment-url-here)

## ✨ Features

- **Smart Predictions**: Predicts delays in 4 categories: No Delay, Regular Delay, Severe Delay, Extreme Delay
- **Real-time Weather**: Integration with wttr.in + Open-Meteo APIs for current weather conditions
- **User-Friendly Interface**: Searchable airport dropdowns with 1000+ airports worldwide
- **Modern Design**: Beautiful dark theme with smooth animations
- **ML Excellence**: Calibrated Random Forest model for accurate probability estimates
- **Fast Performance**: Pre-trained models for instant predictions (no large data loading)

## 🛠️ Tech Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Model**: Calibrated Random Forest Classifier with isotonic calibration
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Weather APIs**: wttr.in, Open-Meteo
- **Data**: Historical flight data + OurAirports database

## 📁 Project Structure

```
flight-delay-predictor/
├── app_new.py                      # Main Flask application
├── create_lightweight_model.py     # Creates deployment-ready model
├── train_model_standalone.py       # Creates full-size model (local use)
├── requirements.txt                # Python dependencies
├── Procfile                       # For deployment
├── .gitignore                     # Git ignore file
├── README.md                      # This file
├── templates/
│   ├── index.html                 # Main prediction interface
│   └── about.html                 # About page
├── utils/
│   └── airport.py                 # Airport data utilities
├── data/
│   ├── airport_codes.json         # Airport mappings
│   └── airports_raw.csv           # Airport database
├── trained_model_lite.pkl         # Lightweight model (40MB - for deployment)
├── airport_encoder_lite.pkl       # Lightweight encoder (50KB)
├── trained_model.pkl              # Full model (4.7GB - local only, gitignored)
└── airport_encoder.pkl            # Full encoder (50KB - local only, gitignored)
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/flight-delay-predictor.git
   cd flight-delay-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **The app includes pre-trained lightweight models** ready for deployment!
   - `trained_model_lite.pkl` (~40MB) - Optimized for free hosting
   - `airport_encoder_lite.pkl` (~50KB) - Airport code encoder

4. **Run the application:**
   ```bash
   python app_new.py
   ```

5. **Open your browser:**
   Navigate to `http://localhost:5000`

### � Model Training (Optional)

If you want to train your own models:

**For production deployment:**
```bash
python create_lightweight_model.py  # Creates ~40MB model
```

**For maximum accuracy (local use only):**
```bash
python train_model_standalone.py    # Creates ~4.7GB model
```

### 🌐 Free Deployment (40MB total size)

Perfect for free hosting platforms:
- **Railway** (recommended)
- **Render** 
- **Vercel**
- **Heroku**

The lightweight model provides excellent performance while fitting comfortably within free tier limits!

## 🤖 Model Performance

- **Algorithm**: Calibrated Random Forest with Isotonic Calibration
- **Features**: Origin airport, destination airport, departure hour, weather conditions
- **Training Data**: 775K+ historical flight records
- **Accuracy**: Calibrated probabilities for reliable uncertainty estimates
- **Speed**: Sub-second predictions using pre-trained models

## 🧠 How It Works

1. **Input Processing**: User selects origin, destination, and departure time
2. **Weather Integration**: Real-time weather data fetched for origin airport
3. **Feature Encoding**: Airports encoded using trained label encoder
4. **ML Prediction**: Calibrated Random Forest model predicts delay probabilities
5. **Result Display**: User-friendly interface shows delay category probabilities

## 👨‍💻 About the Creator

**Cassius Famolaro** - First Generation  
🎓 Student and aspiring aerospace engineer  
🚀 Founder of **Innovation Uprising** - a free online STEM education program for Jamaican students

### Project Background
- Built as part of a machine learning course project
- Combines passion for aviation, engineering, and data science
- Demonstrates real-world applications of predictive modeling
- Explores how delays affect air travel logistics and passenger experience

### Mission
Creating accessible tools that demonstrate the power of data in solving real-world problems while building a foundation for future projects in aerospace, AI, and systems engineering.

## 📄 License

MIT License - feel free to use this project for learning and development!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

For questions about this project or **Innovation Uprising**, feel free to reach out!
