# Intelligent Water Sensor Fault Detection and Quality Monitoring

An **end-to-end machine learning framework** for accurate and reliable water quality monitoring through **sensor fault detection and calibration**. This system transforms raw sensor data into meaningful water quality parameters (such as pH, turbidity, conductivity, and dissolved oxygen) and classifies sensor health status to ensure dependable monitoring. A web-based dashboard provides real-time predictions and actionable insights.

---

## 📖 Abstract
Accurate monitoring of water quality is critical for environmental safety and public health. However, water quality sensors are prone to faults (drift, bias, noise, calibration errors, etc.), leading to unreliable readings.  

This project presents a robust ML pipeline that:
- Calibrates multi-channel raw sensor data into interpretable physical ranges.  
- Classifies sensors as "Good" or "Faulty" using supervised learning techniques.  
- Ensures system reliability by detecting malfunctions before they affect monitoring.  
- Deploys via a user-friendly Flask web interface, enabling real-time analysis and visualization.  

By bridging **sensor engineering, machine learning, and deployment**, this project strengthens resilience in water monitoring systems, supporting proactive water resource management and public health safety.

---

## 🎯 Problem Statement
Water monitoring relies on sensors to measure critical properties such as pH, turbidity, conductivity, and dissolved oxygen. Faulty or drifting sensors can produce misleading data, risking poor environmental management and public health issues.  

This project aims to:
- Build an intelligent system that detects sensor faults automatically.  
- Use machine learning classification to assess sensor health.  
- Improve trustworthiness of water quality monitoring by filtering out faulty sensor data.  
- Provide a real-time dashboard for operators to monitor and take corrective action.

---

## 📊 Dataset
**Source:** [Kaggle – Wafer Sensor Dataset](https://www.kaggle.com/datasets/priyanka369/wafer-sensor-dataset)  
**Data Type:** Multi-channel water quality sensor readings.  

**Parameters Measured:** pH, Turbidity, Conductivity, Dissolved Oxygen, Chlorine Level, Nitrate, Hardness, Temperature, Iron Content, Biochemical Oxygen Demand (BOD).  

**Labels:**  
- Good → Sensor functioning correctly  
- Faulty → Sensor malfunction detected  

This structured dataset enables training ML models to identify faulty sensors and enhance reliability.

---

## ✨ Features
- Sensor Fault Detection – Classifies sensor health as "Good" or "Faulty".  
- Calibration Module – Adjusts raw sensor outputs to realistic ranges.  
- Water Quality Monitoring – Tracks key parameters (pH, turbidity, etc.).  
- Flask Web Dashboard – Real-time monitoring and prediction interface.  
- Notebooks for Analysis – Exploratory Data Analysis (EDA) and ML experiments.  
- Cloud Deployment Ready – Procfile provided for Heroku deployment.  

---

## 🏗️ System Architecture

flowchart TD
    A[Raw Sensor Data] --> B[Calibration Module]
    B --> C[Preprocessed Water Quality Parameters]
    C --> D[ML Fault Detection Model]
    D -->|Good/Faulty| E[Web Dashboard - Flask]
    E --> F[Real-time Insights & Visualization]

---

📂 Project Structure
.
├── artifacts/                   # Stores calibration/model artifacts
├── notebooks/                   # Jupyter notebooks (EDA, experiments)
├── src/                         # ML pipeline, utilities, preprocessing
├── static/css/                  # Frontend styling
├── templates/                   # HTML templates for Flask dashboard
├── application.py               # Main Flask app
├── create_calibration_params.py # Calibration script
├── Water_Sensor_Prediction.csv  # Sample dataset
├── requirements.txt             # Dependencies
├── setup.py                     # Setup configuration
├── Procfile                     # For Heroku deployment
└── .gitignore                   # Git ignore rules
 

---
# 📦 Installation and Setup

To run this project locally:

**Clone the Repository**  
```
git clone https://github.com/siddhivdash/Intelligent-Water-Sensor-Fault-Detection-and-Quality-Monitoring.git
cd Intelligent-Water-Sensor-Fault-Detection-and-Quality-Monitoring
```

**Create Virtual Environment**  
```
python -m venv venv
```

- On Linux/Mac:  
```
source venv/bin/activate
```
- On Windows:  
```
venv\Scripts\activate
```

**Install Dependencies**  
```
pip install -r requirements.txt
```

**Run Calibration (Optional)**  
```
python create_calibration_params.py
```

**Start the Application**  
```
python application.py
```
Now open: [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.


# 🖥️ Usage

- Monitor real-time sensor data via the Flask dashboard.  
- Predict sensor status (Good or Faulty).  
- Upload or process new datasets for analysis.  
- Use calibration script to improve sensor accuracy.  
- Explore Jupyter notebooks for deeper experiments.  


# 🔮 Future Enhancements

Currently, the model performs **binary classification** (Good vs Faulty). Future improvements aim to detect specific fault types for better maintenance strategies, such as:

- **Drift Fault** – Gradual deviation over time.  
- **Stuck-at Fault** – Constant output regardless of conditions.  
- **Bias Fault** – Outputs shifted by a fixed offset.  
- **Spike/Noise Fault** – Sudden random fluctuations.  
- **Communication Fault** – Missing/delayed data due to connectivity.  
- **Calibration Fault** – Systematic error due to miscalibration.  


# 🤝 Contributing

We welcome contributions!  

1. Fork this repo.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m "Added new feature"`).  
4. Push to branch (`git push origin feature-name`).  
5. Submit a Pull Request.  
