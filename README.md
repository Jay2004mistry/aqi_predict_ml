# 🌿 Air Quality Index (AQI) Predictor — India

A machine learning web application that predicts air quality status for Indian cities using live pollutant data.

---

## 🚀 Live Demo

- **Live**: https://jay2004mistry.github.io/aqi_predict_ml/

---

## 📌 Project Overview

This project predicts the **Air Quality Index (AQI) status** (Good, Satisfactory, Moderate, Poor, Very Poor, Severe) for Indian cities based on pollutant readings.

- Trained on **4.3 lakh+ historical pollution records** from 1987–2015 (CPCB India)
- Uses **live data from WAQI API** for real-time predictions
- Deployed as a **FastAPI backend** on Render.com
- Simple **HTML/CSS/JS frontend** hosted on GitHub Pages

---

## 🧠 Machine Learning Pipeline

| Step | Details |
|------|---------|
| Dataset | India Air Quality Data (1987–2015) |
| Records | 4,35,000+ rows |
| Target | AQI Status (6 classes) |
| Features | SO₂, NO₂, RSPM, SPM, State, Location, Type, Year, Month, Season |
| Outlier Treatment | IQR Capping |
| Encoding | Label Encoding (separate encoder per categorical column) |
| Best Model | Random Forest Classifier |
| Train Accuracy | 91.21% |
| Test Accuracy | 90.98% |

---

## 📊 Models Compared

| Model | AUC Score |
|-------|-----------|
| Random Forest | 1.0000 🏆 |
| Decision Tree | 0.9996 |
| KNN | 0.9952 |
| Naive Bayes | 0.9442 |

---

## 🏗️ Project Structure

```
aqi_predict_ml/
├── main.py                 # FastAPI backend
├── aqi_model.pkl           # Trained Random Forest model
├── le_state.pkl            # Label encoder for state
├── le_location.pkl         # Label encoder for location
├── le_type.pkl             # Label encoder for area type
├── feature_columns.json    # Feature column order
├── requirements.txt        # Python dependencies
├── index.html              #frontend ui
└── aqi.ipynb               # Full ML notebook
```

---

## ⚙️ API Endpoints

### `GET /`
Returns API status message.

### `POST /predict`
Predicts AQI category from pollutant readings.

**Request Body:**
```json
{
    "state": "Gujarat",
    "location": "Ahmedabad",
    "type": "Residential, Rural and other Areas",
    "so2": 12.5,
    "no2": 18.3,
    "rspm": 75.0,
    "spm": 140.0,
    "year": 2024,
    "month": 1,
    "season": "Winter"
}
```

**Response:**
```json
{
    "predicted_aqi_category": "Moderate",
    "confidence": 87.50
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn, pandas, numpy |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Live Data | WAQI API |
| Deployment | Render.com (API), GitHub Pages (Frontend) |

---

## 📦 Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/Jay2004mistry/aqi_predict_ml
cd aqi_predict_ml
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run API:**
```bash
uvicorn main:app --reload
```

**4. Open frontend:**
Open `index.html` with Live Server in VS Code.

---

## 📈 AQI Categories

| Status | AQI Range | Health Impact |
|--------|-----------|---------------|
| Good | 0–50 | Safe for all |
| Satisfactory | 51–100 | Minor breathing discomfort for sensitive people |
| Moderate | 101–200 | Breathing discomfort for asthma patients |
| Poor | 201–300 | Breathing discomfort for most people |
| Very Poor | 301–400 | Respiratory illness on prolonged exposure |
| Severe | 401–500 | Affects healthy people, serious impact on sensitive |

---

## 👨‍💻 Author

**Jay Mistry**  
 
[GitHub](https://github.com/Jay2004mistry)

