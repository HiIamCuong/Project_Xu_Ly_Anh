import streamlit as st
import cv2
import numpy as np
from PIL import Image
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9

def app():
    st.title("🧪 Xử lý ảnh số")

    # Mapping chương và các hàm xử lý tương ứng
    CHAPTER_OPS = {
        'Chương 3': {
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
        'Chương 4': {
            'Spectrum': c4.Spectrum,
            'Remove Moire': c4.RemoveMoire,
            'Remove Interference': c4.RemoveInterferenceFilter,
            'Create Motion': c4.CreateMotion,
            'Demotion': c4.DeMotion,
            'Demotion Weiner': lambda x: c4.DeMotion(cv2.medianBlur(x,7))
        },
        'Chương 9': {
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

    # Sidebar chọn ảnh và thao tác
    uploaded = st.sidebar.file_uploader("📂 Chọn ảnh", type=["jpg", "png", "jpeg"])
    chapter = st.sidebar.selectbox("📘 Chọn chương", list(CHAPTER_OPS.keys()))
    op = st.sidebar.selectbox("⚙️ Chọn phép biến đổi", list(CHAPTER_OPS[chapter].keys()))

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Ảnh gốc", use_column_width=True)

        func = CHAPTER_OPS[chapter][op]
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if func in [c3.Negative_Color, c3.HistEqualColor]:
            out = func(img_cv)
        else:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            out_gray = func(gray)
            out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Kết quả ({chapter}): {op}", use_column_width=True)
    else:
        st.info("Vui lòng tải lên ảnh để bắt đầu.")
