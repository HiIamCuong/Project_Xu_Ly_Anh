import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import time

# Streamlit page config
st.set_page_config(page_title="Face Recognition App", layout="wide")

# Sidebar navigation
st.sidebar.image('hcmute.png',width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Trang Chủ", "Nhận diện khuôn mặt"], key="page_nav")

# Load models once and cache
@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create(
        'face_detection_yunet_2023mar.onnx',
        '',
        (320, 320),  # preset input size
        0.9,
        0.3,
        5000
    )
    recognizer = cv2.FaceRecognizerSF.create(
        'face_recognition_sface_2021dec.onnx',
        ''
    )
    svc = joblib.load('svc.pkl')
    label_dict = {0: 'Alice', 1: 'Bob'}
    return detector, recognizer, svc, label_dict

# Initialize models
detector, recognizer, svc, label_dict = load_models()

# Utility: draw annotations
def annotate(frame, faces, names):
    for box, name in zip(faces, names):
        x, y, w, h = map(int, box[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    return frame

# Recognition pipeline
def recognize_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return [], []
    names = []
    for f in faces:
        aligned = recognizer.alignCrop(frame, f)
        feat = recognizer.feature(aligned).reshape(1, -1)
        pred = svc.predict(feat)[0]
        names.append(label_dict.get(pred, "Unknown"))
    return faces.tolist(), names

# Trang Chủ page
if page == "Trang Chủ":
    st.title("Trang Chủ")
    st.write("Welcome to the multi-page Face Recognition App.")
    st.write("Use the sidebar to navigate to the Face Recognition page.")

# Face Recognition page
else:
    st.title("Nhận diện khuôn mặt")
    # Mode selection
    mode = st.sidebar.radio("Input mode:", ["Image", "Video", "Webcam"], key="mode_radio")

    if mode == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img_uploader")
        if uploaded:
            # Read image and convert to BGR for OpenCV models
            img = Image.open(uploaded).convert("RGB")
            frame_rgb = np.array(img)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            faces, names = recognize_frame(frame)
            annotated = annotate(frame.copy(), faces, names)
            # Convert back to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True)

    elif mode == "Video":
        vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="vid_uploader")
        play = st.sidebar.checkbox("Play video", key="play_video")
        if vid_file and play:
            tfile = "temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(vid_file.read())
            cap = cv2.VideoCapture(tfile)
            stframe = st.empty()
            while play and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                faces, names = recognize_frame(frame)
                annotated = annotate(frame, faces, names)
                # Convert BGR to RGB for display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, use_column_width=True)
                time.sleep(0.03)
            cap.release()

    else:  # Webcam
        run = st.sidebar.checkbox("Run Webcam", key="run_webcam")
        frame_placeholder = st.empty()
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break
                    faces, names = recognize_frame(frame)
                    annotated = annotate(frame, faces, names)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, use_column_width=True)
                    time.sleep(0.03)
                cap.release()
