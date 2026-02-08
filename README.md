# AI Powered Waste Segregation and Monitoring System

This project uses YOLOv8 deep learning model trained on the TACO dataset 
to detect and classify waste into categories:

- Organic
- Recyclable
- Other/Landfill

It also integrates IoT-based smart bin monitoring using ultrasonic sensors.

## Features
- Waste detection using YOLOv8
- Streamlit-based web interface
- Smart bin overflow alert system (Arduino + HC-SR04)

## Tech Stack
- Python, YOLOv8, Streamlit
- Arduino Uno, Ultrasonic Sensor
- TACO Dataset

## How to Run

1. Install requirements:

```bash
pip install -r requirements.txt
