import streamlit as st

st.set_page_config(page_title="Ứng dụng xử lý ảnh đa chức năng", layout="wide")

import cv2
import numpy as np
from PIL import Image

# Gọi các module riêng
import face_recognition
import fruit_detection
import image_processing

# ========== 1. CẤU HÌNH CHUNG ==========
st.sidebar.image("hcmute.png", width=100)
st.sidebar.title("📋 Chức năng")

menu = st.sidebar.radio("Chọn chức năng:", [
    "🏠 Trang Chủ", 
    "🍎 Nhận diện trái cây", 
    "😊 Nhận diện khuôn mặt", 
    "🖼️ Xử lý ảnh"
])

# ========== 2. TRANG CHỦ ==========
if menu == "🏠 Trang Chủ":
    st.title("🏠 Ứng dụng xử lý ảnh đa chức năng")
    st.markdown("""
    Xin chào! 👋  
    Ứng dụng này bao gồm:
    - 📸 Nhận diện khuôn mặt từ ảnh, webcam, video.
    - 🍎 Nhận diện trái cây bằng mô hình YOLOv8 ONNX.
    - 🧪 Các phép biến đổi và xử lý ảnh nâng cao theo từng chương.
    Hãy chọn chức năng từ thanh bên trái để bắt đầu.
    """)

# ========== 3. NHẬN DIỆN TRÁI CÂY ==========
elif menu == "🍎 Nhận diện trái cây":
    fruit_detection.app()

# ========== 4. XỬ LÝ ẢNH ==========
elif menu == "🖼️ Xử lý ảnh":
    image_processing.app()

# ========== 5. NHẬN DIỆN KHUÔN MẶT ==========
elif menu == "😊 Nhận diện khuôn mặt":
    face_recognition.app()