import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
from PIL import Image

# 1. Page Configuration for Thesis Portfolio & SEO
st.set_page_config(
    page_title="Deep Litter Poultry Bird Counter | Thesis Project",
    page_icon="🐔",
    layout="wide"
)

st.title("🐔 Deep Litter Poultry Bird Counter")
st.write("""
Automated bird detection and counting system for deep litter housing environments. 
Upload multiple pen images simultaneously from your four-side capture protocol to calculate total bird counts.
""")

# 2. Configure Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_ROBOFLOW_API_KEY"  # Replace with your actual Roboflow API key
)

# 3. Multi-File Image Upload Section
uploaded_files = st.file_uploader(
    "Choose poultry pen images...", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Loaded {len(uploaded_files)} image(s) for batch processing.")
    
    if st.button("Run Batch Bird Detection"):
        with st.spinner("Processing images, enhancing contrast, and counting birds..."):
            for i, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"---")
                st.subheader(f"Image {i+1}: {uploaded_file.name}")
                
                # Read uploaded file into OpenCV format
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                # Apply contrast/brightness enhancement to handle shadows
                enhanced_img = cv2.convertScaleAbs(img_cv, alpha=1.2, beta=20)
                
                # Save temporarily for Roboflow inference SDK
                temp_path = f"temp_enhanced_{i}.jpg"
                cv2.imwrite(temp_path, enhanced_img)
                
                # Run Roboflow Workflow / Model Inference
                result = CLIENT.infer(temp_path, model_id="deep-litter-bird-counter-zgtai/1")
                
                # Extract predictions
                predictions = result.get("predictions", [])
                count = len(predictions)
                
                # Draw bounding boxes on detected birds
                for pred in predictions:
                    x = int(pred["x"] - pred["width"] / 2)
                    y = int(pred["y"] - pred["height"] / 2)
                    w = int(pred["width"])
                    h = int(pred["height"])
                    cv2.rectangle(enhanced_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Convert image back for Streamlit display
                enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                
                # Display metrics and processed image side-by-side or stacked
                st.metric(label=f"Total Birds Counted ({uploaded_file.name})", value=count)
                st.image(enhanced_img_rgb, caption=f"Processed Output: {uploaded_file.name} ({count} birds)", use_column_width=True)
