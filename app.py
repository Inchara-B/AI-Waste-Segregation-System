import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from collections import Counter # <-- ADDED FOR FREQUENCY COUNT

# --- CONFIGURATION & SEGREGATION LOGIC ---

# 1. Define the 60-class mapping to your final 3 categories 
WASTE_MAP = {
    'Organic': [
        'Food waste'
    ],
    
    'Recyclable': [
        # --- METAL ---
        'Aluminium foil', 
        'Aluminium blister pack', 
        'Metal bottle cap', 
        'Food Can', 
        'Aerosol', 
        'Drink can',
        'Scrap metal',
        'Pop tab',
        
        # --- GLASS ---
        'Glass bottle', 
        'Broken glass', 
        'Glass jar', 
        'Glass cup',
        
        # --- PLASTIC ---
        'Other plastic bottle', 
        'Clear plastic bottle', 
        'Plastic bottle cap',
        'Other plastic cup', 
        'Plastic lid', 
        'Other plastic', 
        'Plastic film', 
        'Six pack rings',
        'Garbage bag', 
        'Other plastic wrapper', 
        'Single-use carrier bag',
        'Polypropylene bag', 
        'Spread tub', 
        'Tupperware', 
        'Other plastic container', 
        'Plastic glooves',
        'Plastic utensils', 
        'Squeezable tube', 
        'Plastic straw',
        'Disposable plastic cup',
        'Disposable food container',
        
        # --- PAPER/CARDBOARD ---
        'Carded blister pack', 
        'Toilet tube', 
        'Other carton', 
        'Egg carton', 
        'Drink carton',
        'Corrugated carton', 
        'Meal carton', 
        'Pizza box', 
        'Paper cup',
        'Magazine paper', 
        'Tissues', 
        'Wrapping paper', 
        'Normal paper', 
        'Paper bag',
        'Plastified paper bag', 
        'Paper straw'
    ],
    
    'Other/Landfill': [
        # --- NON-RECYCLABLE / HAZARDOUS ---
        'Battery', 
        'Crisp packet',         
        'Foam cup',             
        'Foam food container',  
        'Rope & strings',       
        'Shoe',                 
        'Styrofoam piece',      
        'Unlabeled litter',     
        'Cigarette'
    ]
}

# Invert the map for quick category lookup
CLASS_TO_CATEGORY = {cls: cat for cat, classes in WASTE_MAP.items() for cls in classes}

# Define segregation function (UPDATED FOR FREQUENCY)
def segregate_waste(detected_classes):
    """Categorizes a list of detected items and counts frequency."""
    if not detected_classes:
        return "No waste items were clearly detected.", []
    
    segregated = {'Organic': 0, 'Recyclable': 0, 'Other/Landfill': 0}
    
    # 1. Count how many of each item were detected (e.g., {'Paper cup': 3, 'Aerosol': 1})
    item_frequency = Counter(detected_classes)

    # 2. Map frequency to category and calculate total category count
    item_breakdown = []
    
    for cls, count in item_frequency.items():
        # Use the global lookup map
        category = CLASS_TO_CATEGORY.get(cls, 'Other/Landfill') 
        segregated[category] += count
        
        # Create a detailed list for the table: (Count, Item, Category)
        item_breakdown.append({
            'Count': count, 
            'Item Detected': cls, 
            'Category': category
        })
        
    summary = (
        f"Detection complete. Found **{segregated.get('Organic', 0)} Organic**, "
        f"**{segregated.get('Recyclable', 0)} Recyclable**, and "
        f"**{segregated.get('Other/Landfill', 0)} Other/Landfill** item(s)."
    )
    return summary, item_breakdown # Returns a list of dictionaries for the table


# --- MODEL LOADING & ERROR HANDLING ---

st.set_page_config(layout="wide", page_title="AI Waste Segregation System")

st.title("♻️ AI Waste Segregation & Monitoring System")
st.write("Upload an image of waste to instantly classify and segregate items.")
st.divider()

# Check for model file integrity early
if not os.path.exists('best.pt'):
    st.error("FATAL ERROR: Model file 'best.pt' not found. Please ensure it is in the same folder as app.py.")
    st.stop()
    
# Load the custom-trained model
try:
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')

    model = load_model()

except Exception as e:
    st.error(f"FATAL ERROR: Failed to initialize YOLO model. Check dependency installations.")
    st.error(f"Error Details: {e}")
    st.stop()


# --- STREAMLIT FRONTEND & INFERENCE ---

st.success("AI Model (best.pt) loaded successfully!") 

uploaded_file = st.file_uploader("Choose an image of waste items...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Image to be processed.", use_column_width=True)

    with col2:
        st.subheader("AI Analysis & Segregation")
        
        with st.spinner('The trained model is analyzing the image...'):
            # --- CRITICAL INFERENCE LINE (using conf=0.25 fix) ---
            # New: Enable Test-Time Augmentation
            results = model(image, conf=0.25, iou=0.85, augment=True)
            
            # Get class names of detected objects
            detected_classes = []
            if results and results[0].boxes:
                detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
            
            # Segregate waste based on the mapping
            summary, item_list = segregate_waste(detected_classes)
            
            # Plot results on the image and convert BGR (YOLO output) to RGB (display)
            annotated_image = results[0].plot() 
            annotated_image_rgb = Image.fromarray(annotated_image[..., ::-1].astype(np.uint8)) 
            
            st.image(annotated_image_rgb, caption='Processed Image with Bounding Boxes.', use_column_width=True)
            
            st.success(summary, icon="🗑️")
            
            if item_list:
                st.write("**Segregation Breakdown:**")
                # Use st.dataframe to display the list of dictionaries (frequency table)
                st.dataframe(item_list, use_container_width=True, hide_index=True)