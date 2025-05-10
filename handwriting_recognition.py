import streamlit as st
from PIL import Image
import numpy as np
import easyocr

# Khởi tạo Reader một lần duy nhất để tái sử dụng
# ['en','vi'] cho phép nhận diện cả tiếng Anh và tiếng Việt
reader = easyocr.Reader(['en', 'vi'], gpu=False)  

def app():
    st.title("✍️ Nhận diện chữ viết tay (EasyOCR)")
    uploaded_file = st.file_uploader("Tải lên ảnh chữ viết tay", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    # Đọc và hiển thị ảnh
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh chữ viết tay", use_column_width=True)

    # Chuyển sang numpy array để feed cho EasyOCR
    img_array = np.array(image)

    # Gọi EasyOCR để nhận diện văn bản
    results = reader.readtext(img_array, detail=0, paragraph=True)

    # Hiển thị kết quả
    st.subheader("📄 Kết quả nhận diện:")
    # join các đoạn văn bản thành 1 chuỗi
    text = "\n".join(results)
    st.text_area("Văn bản thu được:", text, height=200)