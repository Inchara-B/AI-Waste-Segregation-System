# ♻️ AI Powered Waste Segregation and Monitoring System for Smart Cities

This project combines **AI + IoT** for smart waste management.

---

## 📌 Overview

- Waste detection using **YOLOv8**
- Streamlit dashboard for classification
- Smart bin monitoring using ultrasonic sensor + buzzer

---

## 🚀 Features

- Detects waste into Organic / Recyclable / Other categories  
- Real-time bin fill-level monitoring  
- Overflow alert when bin exceeds 80%

---

## 🛠️ Tech Stack

- Python, YOLOv8 (Ultralytics)
- Streamlit, OpenCV
- Arduino Uno, HC-SR04 Sensor

---

## 📂 Project Files

- `app.py` → Streamlit application  
- `best.pt` → Trained YOLOv8 model  
- `requirements.txt` → Dependencies  
- `images/` → Output screenshots  

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open in browser:

http://localhost:8501

🎥 Hardware Demo: https://drive.google.com/file/d/1v44m-xrsctaaFlzyT0W-0_w-D3qzgeCy/view?usp=drive_link

