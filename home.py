import streamlit as st

st.set_page_config(page_title="Ứng dụng xử lý ảnh đa chức năng", layout="wide")

import cv2
import numpy as np
from PIL import Image

# Gọi các module riêng
import face_recognition
import fruit_detection
import image_processing
import garbage_detection
import handwriting_recognition

# ========== 1. CẤU HÌNH CHUNG ==========
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("hcmute.png", width=120)
st.sidebar.title("📋 Chức năng")

menu = st.sidebar.radio("Chọn chức năng:", [
    "🏠 Trang Chủ", 
    "😊 Nhận diện khuôn mặt", 
    "🍎 Nhận dạng trái cây", 
    "🖼️ Xử lý ảnh",
    "🗑️ Phân loại rác thải",
    "✍️ Nhận diện chữ viết tay"

])

# ========== 2. TRANG CHỦ ==========
if menu == "🏠 Trang Chủ":
    st.title("🏠 Ứng dụng xử lý ảnh đa chức năng")
    st.markdown("""
    Xin chào! 👋  

    Đây là ứng dụng xử lý ảnh đa chức năng với các tính năng nổi bật:

    - 😊 **Nhận diện khuôn mặt**  
      → Hỗ trợ nhận diện **5 người dùng** từ ảnh, webcam hoặc video.
    
    - 🍎 **Nhận dạng trái cây** sử dụng mô hình YOLOv8 (ONNX)  
      → Nhận diện **5 loại quả**: **Bưởi, Sầu riêng, Táo, Thanh long, Thơm**.

    - 🖼️ **Xử lý ảnh nâng cao**  
      → Bao gồm các phép biến đổi từ **Chương 3, 4 và 9** như: âm bản, lọc nhiễu, biến đổi Fourier, phân đoạn ảnh...

    - 🗑️ **Phân loại rác thải**  
      → Phân loại ảnh rác thành các nhóm **hữu cơ**, **tái chế** bằng mô hình YOLOv8 (ONNX).

    ---

    👨‍💻 **Thành viên thực hiện:**
    - **Lê Nhựt Anh** – MSSV: `22110279`
    - **Nguyễn Sỹ Cường** – MSSV: `22133007`

    ---  
    👉 Vui lòng chọn chức năng ở thanh bên trái để bắt đầu trải nghiệm!
    """)

# ========== 3. NHẬN DIỆN TRÁI CÂY ==========
elif menu == "🍎 Nhận dạng trái cây":
    fruit_detection.app()

# ========== 4. XỬ LÝ ẢNH ==========
elif menu == "🖼️ Xử lý ảnh":
    image_processing.app()

# ========== 5. NHẬN DIỆN KHUÔN MẶT ==========
elif menu == "😊 Nhận diện khuôn mặt":
    face_recognition.app()

elif menu == "🗑️ Phân loại rác thải":
    garbage_detection.app()

elif menu == "✍️ Nhận diện chữ viết tay":
    handwriting_recognition.app()