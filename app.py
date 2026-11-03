import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import numpy as np
import cv2

st.title("🐔 Poultry Counting Web App (Multi-Image)")

# Initialize the Roboflow HTTP client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="BrgrhWWJhrjyx7RgkiEd"
)

# Enable multiple file uploads using accept_multiple_files=True
uploaded_files = st.file_uploader("Upload multiple poultry pen images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Run Batch Bird Count"):
        with st.spinner("Analyzing all images with Roboflow..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"---")
                st.subheader(f"Image {idx + 1}: {uploaded_file.name}")
                
                # Save each file temporarily
                temp_filename = f"temp_image_{idx}.jpg"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Run your specific workflow for each image
                result = CLIENT.run_workflow(
                    workspace_name="dhineshkumars-workspace",
                    workflow_id="deep-litter-bird-counter-vdeep-litter-bird-counter-zgtai-1-rfdetr-small-t1-logic",
                    images={"image": temp_filename}
                )
                
                try:
                    predictions = result[0]["predictions"]["predictions"]
                    total_count = len(predictions)
                    
                    st.metric(label=f"Birds Counted in {uploaded_file.name}", value=total_count)
                    
                    # Load image with OpenCV to draw bounding boxes
                    img = cv2.imread(temp_filename)
                    
                    for pred in predictions:
                        x = int(pred["x"])
                        y = int(pred["y"])
                        w = int(pred["width"])
                        h = int(pred["height"])
                        
                        x1 = int(x - (w / 2))
                        y1 = int(y - (h / 2))
                        x2 = int(x + (w / 2))
                        y2 = int(y + (h / 2))
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption=f"Detected Birds: {total_count}", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error parsing results for {uploaded_file.name}: {e}")
                    st.write(result)