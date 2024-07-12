import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import requests
import os

# Function to download YOLOv8 weights
def download_weights(url, dest_path):
    with open(dest_path, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)

# Function to process video and detect traffic lights
def process_video(video_path, model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Define a mapping from class names to colors
    color_map = {
        'red': (0, 0, 255),       # Red color
        'green': (0, 255, 0),     # Green color
        'yellow': (0, 255, 255),  # Yellow color
        'off': (128, 128, 128)    # Gray color
    }

    # Open video file or capture from camera
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file. Please check the file path and format.")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object to save the video
    out_path = 'output_video.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Annotate the frame
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]
            cls = detection.cls[0]
            label = model.names[int(cls)]

            # Determine the bounding box color based on the label
            bbox_color = color_map.get(label, (255, 255, 255))  # Default to white if label not in color_map

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)

            # Put label
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

        # Write the frame to the output video
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()

    return out_path

# Streamlit UI
st.title('Traffic Light Detection')
st.write("Upload a video file to detect traffic lights")

# Upload video file
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Download YOLOv8 weights
    weights_url = 'https://www.dropbox.com/scl/fi/4ez66ikuuni8glvpyy8bp/best.pt?rlkey=vjet63pwakuiw5j2omnrhsui6&st=jhfvt7f3&dl=0'  # Replace with your actual URL
    weights_path = 'best.pt'
    if not os.path.exists(weights_path):
        download_weights(weights_url, weights_path)

    # Process video
    output_video_path = process_video(video_path, weights_path)
    if output_video_path:
        st.success("Video processing complete. Download the output video below:")

        # Display and download output video
        with open(output_video_path, 'rb') as video_file:
            st.download_button(label="Download Output Video", data=video_file, file_name="output_video.mp4")
        st.video(output_video_path)
    else:
        st.error("Failed to process video.")
