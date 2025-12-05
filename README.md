# 📱 Phone Price Predictor - AI Chatbot

A modern, AI-powered chatbot GUI that predicts whether a phone is **Expensive** or **Non-Expensive** based on its specifications using a Random Forest machine learning model.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Input Fields](#-input-fields)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- 🎨 **Modern Dark Theme** - Sleek, professional UI design
- 💬 **Chat-Style Interface** - Intuitive prediction history display
- 🤖 **AI-Powered Predictions** - Random Forest model with ~85%+ accuracy
- 📊 **Confidence Scores** - Shows prediction confidence percentage
- 🔄 **Auto Model Training** - Model trains automatically on startup
- 📱 **Comprehensive Inputs** - All phone specifications supported

---

## 📦 Requirements

### Python Version
- Python **3.8** or higher

### Required Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Machine learning (Random Forest) |
| `tkinter` | GUI framework (included with Python) |

---

## 🚀 Installation

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd Spec-to-Price-main

# Or simply download and extract the ZIP file
```

### Step 2: Install Python Dependencies

```bash
# Using pip
pip install pandas numpy scikit-learn

# Or using requirements (if available)
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas; import numpy; import sklearn; print('All dependencies installed!')"
```

---

## 🎮 Usage

### Running the Application

```bash
cd Spec-to-Price-main
python chatbot_gui.py
```

### Quick Start Guide

1. **Launch** - Run `python chatbot_gui.py`
2. **Wait** - Model loads automatically (few seconds)
3. **Fill** - Enter phone specifications in the left panel
4. **Predict** - Click "🔮 Predict Price Category"
5. **View** - See result in the chat panel (right side)

---

## 📝 Input Fields

### Binary Options (Toggle On/Off)
| Field | Description |
|-------|-------------|
| Dual SIM | Dual SIM card support |
| 4G | 4G LTE connectivity |
| 5G | 5G connectivity |
| Vo5G | Voice over 5G |
| NFC | Near Field Communication |
| IR Blaster | Infrared blaster |
| Memory Card | SD card support |

### Brand & Processor
| Field | Example Values |
|-------|----------------|
| Brand | Samsung, Apple, Xiaomi, OnePlus... |
| Processor Brand | Snapdragon, MediaTek, Exynos, Apple... |
| Processor Series | 870, 8 Gen 2, A17 Pro... |
| Rating | 0-100 (user rating) |

### Display
| Field | Example Values |
|-------|----------------|
| Screen Size | 6.5 (inches) |
| Resolution Width | 1080 (pixels) |
| Resolution Height | 2400 (pixels) |
| Refresh Rate | 120 (Hz) |
| Notch Type | Punch Hole, Water Drop, No Notch... |

### Performance
| Field | Example Values |
|-------|----------------|
| CPU Cores | 8 |
| Clock Speed | 3.0 (GHz) |
| RAM | 8 (GB) |
| Storage | 256 (GB) |

### Camera
| Field | Example Values |
|-------|----------------|
| Main Camera | 48 (MP) |
| Rear Cameras Count | 3 |
| Front Camera | 16 (MP) |
| Front Cameras Count | 1 |

### Battery & Storage
| Field | Example Values |
|-------|----------------|
| Battery | 5000 (mAh) |
| Fast Charging | 65 (Watts) |
| Max SD Card | 512 (GB) |

### Operating System
| Field | Example Values |
|-------|----------------|
| OS | Android, iOS, HarmonyOS |
| Version | v14, v17, etc. |

---

## 📁 Project Structure

```
Spec-to-Price-main/
├── chatbot_gui.py      # Main GUI application
├── preprocessing.py    # Data preprocessing functions
├── Random Forest       # Random Forest model script
├── KNN                 # KNN model script
├── Logistic-Regression # Logistic Regression script
├── Datasets/
│   ├── train.csv      # Training dataset
│   └── test.csv       # Test dataset
└── README.md          # This documentation
```

---

## ⚙️ How It Works

1. **Data Loading** - Loads `train.csv` from the Datasets folder
2. **Preprocessing** - Encodes categorical variables, handles missing values
3. **Model Training** - Trains Random Forest with 200 trees
4. **Prediction** - Takes user inputs, preprocesses, and predicts

### Model Parameters
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'pandas'` | Run `pip install pandas` |
| `ModuleNotFoundError: No module named 'sklearn'` | Run `pip install scikit-learn` |
| GUI doesn't open | Ensure `tkinter` is installed with Python |
| Model loading fails | Check that `Datasets/train.csv` exists |

### Windows-Specific
```bash
# If tkinter is not installed
pip install tk
```

### Linux-Specific
```bash
# Install tkinter on Ubuntu/Debian
sudo apt-get install python3-tk
```

### macOS-Specific
```bash
# tkinter usually comes with Python on macOS
# If not, reinstall Python from python.org
```

---

## 📄 License

This project is for educational purposes.

---

## 🤝 Contributing

Feel free to submit issues and pull requests!

---

Made with ❤️ using Python & Tkinter
