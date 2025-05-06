import streamlit as st
import cv2
import joblib
import time
import os
import numpy as np
from PIL import Image

@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', '', (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', '')
    svc = joblib.load('svc.pkl')
    labels = ["HongNhung", "KhanhHuy", "KimLoi", "NhutAnh", "SyCuong"]
    return detector, recognizer, svc, labels

def annotate(frame, faces, names):
    for box, name in zip(faces, names):
        x, y, w, h = map(int, box[:4])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def recognize_frame(frame, detector, recognizer, svc, labels):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return [], []
    names = []
    for f in faces:
        aligned = recognizer.alignCrop(frame, f)
        feat = recognizer.feature(aligned).reshape(1, -1)
        p = svc.predict(feat)[0]
        names.append(labels[p] if 0 <= p < len(labels) else 'Unknown')
    return faces.tolist(), names

def app():
    st.title("😊 Nhận diện khuôn mặt từ ảnh/video/webcam")
    detector, recognizer, svc, labels = load_models()
    mode = st.sidebar.radio("Chế độ nhập", ["Ảnh", "Video", "Webcam"])

    if mode == "Ảnh":
        file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces, names = recognize_frame(frame, detector, recognizer, svc, labels)
            ann = annotate(frame.copy(), faces, names)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)

    elif mode == "Video":
        vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
        play = st.sidebar.checkbox("Phát video")
        if vid and play:
            path = "temp.mp4"
            with open(path, "wb") as f: f.write(vid.read())
            cap = cv2.VideoCapture(path)
            ph = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                faces, names = recognize_frame(frame, detector, recognizer, svc, labels)
                ann = annotate(frame, faces, names)
                ph.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)
                time.sleep(0.03)
            cap.release()
            os.remove(path)

    else:
        start = st.sidebar.button("Bật Webcam")
        stop = st.sidebar.button("Dừng Webcam")
        ph2 = st.empty()
        if start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Không thể truy cập webcam.")
            run = True
            while run:
                ret, fr = cap.read()
                if not ret: break
                faces, names = recognize_frame(fr, detector, recognizer, svc, labels)
                ann = annotate(fr, faces, names)
                ph2.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.03)
                if stop: run = False
            cap.release()
