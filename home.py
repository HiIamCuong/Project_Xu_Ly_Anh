import io
from typing import Any

import streamlit as st
from PIL import Image
import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    """
    A class to perform object detection inference using Ultralytics YOLO.
    """
    def __init__(self, model: str = None):
        check_requirements("streamlit>=1.29.0")
        self.st = st
        self.model_path = model
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame = None
        self.ann_frame = None
        self.vid_file_name = None
        self.selected_ind = []
        self.model = None
        LOGGER.info(f"Ultralytics Solutions: ✅ model={self.model_path}")

    def web_ui(self):
        menu_style = "<style>MainMenu {visibility: hidden;}</style>"
        title = ("<h1 style='color:#FF64DA; text-align:center;'>"
                 "Ultralytics YOLO Streamlit Application</h1>")
        subtitle = ("<h4 style='color:#042AFF; text-align:center;'>"
                    "Real-time object detection with Ultralytics YOLO! 🚀</h4>")
        #self.st.set_page_config(page_title="YOLO App", layout="wide")
        self.st.markdown(menu_style, unsafe_allow_html=True)
        self.st.markdown(title, unsafe_allow_html=True)
        self.st.markdown(subtitle, unsafe_allow_html=True)        

    def detectobject(self):
        self.st.title("Inference Settings")
        self.source = self.st.selectbox("Video Source", ("webcam", "video"))
        self.enable_trk = self.st.radio("Enable Tracking", ("Yes", "No"))
        self.conf = float(self.st.slider("Confidence", 0.0, 1.0, self.conf, 0.01))
        self.iou = float(self.st.slider("IoU", 0.0, 1.0, self.iou, 0.01))
        if self.source == "video":
            vid = self.st.file_uploader("Upload Video", type=["mp4","mov","avi","mkv"])
            if vid:
                with open("ultralytics.mp4","wb") as f: f.write(vid.read())
                self.vid_file_name = "ultralytics.mp4"
        else:
            self.vid_file_name = 0
        self.st.markdown("---")
        self.st.subheader("Model & Classes")
        models = [x.replace("yolo","YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path: models.insert(0, self.model_path.split('.pt')[0])
        sel = self.st.selectbox("Model", models)
        with self.st.spinner("Loading model..."):
            self.model = YOLO(f"{sel.lower()}.pt")
        self.st.success("Model loaded")
        names = list(self.model.names.values())
        cls = self.st.multiselect("Classes", names, default=names[:3])
        self.selected_ind = [names.index(c) for c in cls]
        col1, col2 = self.st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()
        if not st.session_state.get("camera_on", False) and self.st.button("Start camera"):
            st.session_state.camera_on = True
        if self.st.button("Stop Camera"):
            st.session_state.camera_on = False
        # Inference loop
        if st.session_state.get("camera_on", False):
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("Cannot open source")
                return
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    break
                results = (
                    self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True)
                    if self.enable_trk == "Yes" else
                    self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                )
                ann = results[0].plot()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.org_frame.image(rgb, channels="RGB")
                self.ann_frame.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()
        cv2.destroyAllWindows()
        # Start/Stop buttons in sidebar

# --- Main App ---
# Page config and sidebar navigation
#st.set_page_config(page_title="Streamlit Multi-Page YOLO App", layout="wide", initial_sidebar_state="expanded")
# Logo in sidebar
Image.open("default.png")
st.sidebar.image("default.png", use_column_width=True)
# Navigation
if "page" not in st.session_state:
    st.session_state.page = "Trang Chủ"
st.sidebar.markdown("---")
for p in ["Trang Chủ", "Nhận diện vật thể"]:
    if st.sidebar.button(p):
        st.session_state.page = p
# Content
if st.session_state.page == "Trang Chủ":
    st.title("Trang Chủ")
    st.write("Welcome to the multi-page app.")
elif st.session_state.page == "Nhận diện vật thể":
    st.title("Giới Thiệu & YOLO Inference")
    inf = Inference()
    inf.detectobject()
