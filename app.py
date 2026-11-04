import base64
import io
from PIL import Image, ImageDraw
import requests
import streamlit as st

st.set_page_config(
    page_title="Poultry Bird Counter", page_icon="🐔", layout="centered"
)

st.title("🐔 Poultry Bird Counter")
st.write(
    "Upload your pen images below to detect and count birds automatically."
)

# Sidebar configuration for Roboflow
st.sidebar.header("Roboflow Settings")
api_key = st.sidebar.text_input("API Key", type="password")
project_id = st.sidebar.text_input(
    "Project ID", value="poultry-counter"
)  # Update with your project ID
version = st.sidebar.text_input("Version Number", value="1")

uploaded_files = st.file_uploader(
    "Upload pen photos (multi-image batch)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
  if not api_key:
    st.warning("Please enter your Roboflow API Key in the sidebar to proceed.")
  else:
    for uploaded_file in uploaded_files:
      image = Image.open(uploaded_file).convert("RGB")

      # Convert image to base64 for Roboflow HTTP API
      buffered = io.BytesIO()
      image.save(buffered, format="JPEG")
      img_str = base64.b64encode(buffered.getvalue()).decode("ascii")

      # Call Roboflow API directly (bypasses broken SDK/cv2 dependencies)
      upload_url = (
          f"https://detect.roboflow.com/{project_id}/{version}?api_key={api_key}"
      )

      with st.spinner(f"Analyzing {uploaded_file.name}..."):
        response = requests.post(
            upload_url,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

      if response.status_code == 200:
        result = response.json()
        predictions = result.get("predictions", [])
        bird_count = len(predictions)

        st.metric(
            label=f"Total Birds in {uploaded_file.name}", value=bird_count
        )

        # Draw bounding boxes on image
        draw = ImageDraw.Draw(image)
        for box in predictions:
          x, y, w, h = box["x"], box["y"], box["width"], box["height"]
          left = x - w / 2
          top = y - h / 2
          right = x + w / 2
          bottom = y + h / 2
          draw.rectangle([left, top, right, bottom], outline="#00FF00", width=4)

        st.image(
            image,
            caption=f"Processed: {uploaded_file.name} ({bird_count} birds)",
            use_container_width=True,
        )
        st.divider()
      else:
        st.error(
            f"API Error for {uploaded_file.name}:"
            f" {response.json().get('message', response.text)}"
        )
