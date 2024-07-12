import streamlit as st
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('/content/drive/MyDrive/Yolov8s20e/runs2/weights/best.pt')

# Define a mapping from class names to colors
color_map = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'off': (128, 128, 128)
}

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('/content/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for detection in results[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]
            cls = detection.cls[0]
            label = model.names[int(cls)]
            bbox_color = color_map.get(label, (255, 255, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)
        out.write(frame)

    cap.release()
    out.release()
    return '/content/output_video.mp4'

# Streamlit app
st.title('Traffic Light Detection')
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
if uploaded_file is not None:
    video_path = uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video(video_path)
    output_video_path = process_video(video_path)
    st.video(output_video_path)
