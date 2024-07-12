import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')  # Ensure you have the correct path to the YOLOv8 weights

# Define a mapping from class names to colors
color_map = {
    'red': (0, 0, 255),       # Red color
    'green': (0, 255, 0),     # Green color
    'yellow': (0, 255, 255),  # Yellow color
    'off': (128, 128, 128)    # Gray color
}

# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Use a temporary file to save the output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = temp_output.name

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

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
    cv2.destroyAllWindows()

    return output_video_path

# Streamlit app
st.title('Traffic Light Detection')
uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        video_path = temp_input.name

    st.video(video_path)
    output_video_path = process_video(video_path)
    st.video(output_video_path)
