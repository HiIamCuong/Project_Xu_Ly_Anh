import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import time
import os
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9

# Mapping operations by chapter
CHAPTER_OPS = {
    'Chapter3': {
        'Negative': c3.Negative,
        'Negative Color': c3.Negative_Color,
        'Logarit': c3.Logarit,
        'Power': c3.Power,
        'PiecewiseLinear': c3.PiecewiseLinear,
        'Histogram': c3.Histogram,
        'Hist Equal': lambda x: cv2.equalizeHist(x),
        'Hist Equal Color': c3.HistEqualColor,
        'Local Hist': c3.LocalHist,
        'Hist Stat': c3.HistStat,
        'Smooth Box': lambda x: cv2.boxFilter(x, cv2.CV_8UC1, (21,21)),
        'Smooth Gauss': lambda x: cv2.GaussianBlur(x, (43,43), 7.0),
        'Median Filter': lambda x: cv2.medianBlur(x, 5),
        'Create Impulse Noise': c3.CreateImpulseNoise,
        'Sharp': c3.Sharp
    },
    'Chapter4': {
        'Spectrum': c4.Spectrum,
        'Remove Moire': c4.RemoveMoire,
        'Remove Interference': c4.RemoveInterferenceFilter,
        'Create Motion': c4.CreateMotion,
        'Demotion': c4.DeMotion,
        'Demotion Weiner': lambda x: c4.DeMotion(cv2.medianBlur(x,7))
    },
    'Chapter9': {
        'Erosion': c9.Erosion,
        'Dilation': c9.Dilation,
        'Boundary': c9.Boundary,
        'Contour': c9.Contour,
        'Convex Hull': c9.ConvexHull,
        'Defect Detect': c9.DefectDetect,
        'Connected Components': c9.ConnectedComponents,
        'Remove Small Rice': c9.RemoveSmallRice
    }
}

# Apply selected operation
def apply_operation(img, func):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # color ops
    if func in [c3.Negative_Color, c3.HistEqualColor]:
        out = func(img_cv)
    else:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        out_gray = func(gray)
        out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# Face recognition helpers
@st.cache_resource
def load_models():
    detector = cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx','',(320,320),0.9,0.3,5000)
    recognizer = cv2.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx','')
    svc = joblib.load('svc.pkl')
    labels = ["HongNhung","KhanhHuy","KimLoi","NhutAnh","SyCuong"]
    return detector, recognizer, svc, labels

def annotate(frame, faces, names):
    for box, name in zip(faces, names):
        x,y,w,h = map(int, box[:4])
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return frame

def recognize_frame(frame, detector, recognizer, svc, labels):
    h,w = frame.shape[:2]
    detector.setInputSize((w,h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces)==0:
        return [], []
    names = []
    for f in faces:
        aligned = recognizer.alignCrop(frame, f)
        feat = recognizer.feature(aligned).reshape(1, -1)
        p = svc.predict(feat)[0]
        names.append(labels[p] if 0<=p<len(labels) else 'Unknown')
    return faces.tolist(), names

# Streamlit config
st.set_page_config(page_title="Multi-App Streamlit", layout="wide")
st.sidebar.image('hcmute.png', width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Chọn chức năng", ["Trang Chủ", "Nhận diện khuôn mặt", "Xử lý ảnh"])

if page == "Trang Chủ":
    st.title("Trang Chủ")
    st.write("Chọn một chức năng từ thanh bên.")

elif page == "Xử lý ảnh":
    st.title("Ứng dụng Xử Lý Ảnh Số")
    uploaded = st.sidebar.file_uploader("Chọn ảnh", type=["jpg","png","jpeg"])
    chapter = st.sidebar.selectbox("Chọn chương", list(CHAPTER_OPS.keys()))
    ops = CHAPTER_OPS[chapter]
    op = st.sidebar.selectbox("Chọn phép biến đổi", list(ops.keys()))
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Ảnh gốc", use_column_width=True)
        with st.spinner('Đang xử lý...'):
            out = apply_operation(img, ops[op])
        st.image(out, caption=f"Kết quả ({chapter}): {op}", use_column_width=True)
    else:
        st.info("Vui lòng tải lên ảnh để bắt đầu.")

else:
    st.title("Nhận diện khuôn mặt")
    detector, recognizer, svc, labels = load_models()
    mode = st.sidebar.radio("Chế độ nhập", ["Ảnh", "Video", "Webcam"])
    if mode == "Ảnh":
        file = st.file_uploader("Upload ảnh", type=["jpg","jpeg","png"])
        if file:
            img = Image.open(file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces, names = recognize_frame(frame, detector, recognizer, svc, labels)
            ann = annotate(frame.copy(), faces, names)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)
    elif mode == "Video":
        vid = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"])
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
            if not cap.isOpened(): st.error("Không thể truy cập webcam.")
            run = True
            while run:
                ret, fr = cap.read()
                if not ret: break
                faces, names = recognize_frame(fr, detector, recognizer, svc, labels)
                ann = annotate(fr, faces, names)
                ph2.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)
                time.sleep(0.03)
                if stop: run = False
            cap.release()