import streamlit as st
import cv2
import numpy as np
from PIL import Image
import Chapter3 as c3
import Chapter4 as c4
import Chapter9 as c9
import io

def app():
    st.title("🖼️ Xử lý ảnh")

    # Khai báo phép xử lý
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
            'Smooth Box': lambda x: cv2.boxFilter(x, cv2.CV_8UC1, (21, 21)),
            'Smooth Gauss': lambda x: cv2.GaussianBlur(x, (43, 43), 7.0),
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
            'Demotion Weiner': lambda x: c4.DeMotion(cv2.medianBlur(x, 7))
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

    # Ảnh mặc định theo chương và mục
    DEFAULT_IMAGES = {
        'Chương 3': {
            'Negative': 'images/chapter3/1_Negative_Image.tif',
            'Negative Color': 'images/chapter3/2_Negative_Color.tif',
            'Logarit': 'images/chapter3/3_Logarit.tif',
            'PiecewiseLinear': 'images/chapter3/5_PiecewiseLinear.png',
            'Histogram': 'images/chapter3/6_Histogram.tif',
            'Hist Equal': 'images/chapter3/7_Histogram_Equal.png',
            'Hist Equal Color': 'images/chapter3/8_Histogram_Equal_Color.tif',
            'Local Hist': 'images/chapter3/9_Local_Histogram.tif',
            'Hist Stat': 'images/chapter3/10_Histogram_statistic.tif',
            'Smooth Box': 'images/chapter3/11_Smooth_box.tif',
            'Smooth Gauss': 'images/chapter3/12_Smooth_gauss.tif',
            'Median Filter': 'images/chapter3/13_Median_filter.tif',
            'Sharp': 'images/chapter3/14_Sharpening.tif'
        },
        'Chương 4': {
            'Spectrum': 'images/chapter4/1_Spectrum.tif',
            'Remove Moire': 'images/chapter4/2_Remove_moire.tif',
            'Remove Interference': 'images/chapter4/3_Remove_interference.tif',
            'Create Motion': 'images/chapter4/4_Create_motion.tif',
            'Demotion': 'images/chapter4/5_Demotion.tif',
            'Demotion Weiner': 'images/chapter4/6_Demotion_noise.tif'
        },
        'Chương 9': {
            'Erosion': 'images/chapter9/1_Erosion.tif',
            'Dilation': 'images/chapter9/2_Dilation.tif',
            'Boundary': 'images/chapter9/3_Boundary.tif',
            'Contour': 'images/chapter9/4_Contour.tif',
            'Connected Components': 'images/chapter9/5_Connected_Components.tif',
            'Remove Small Rice': 'images/chapter9/6_Remove_Small_Rice.tif'
        }
    }

    # Sidebar: chọn ảnh, chương và phép biến đổi
    uploaded = st.sidebar.file_uploader("📂 Chọn ảnh", type=["jpg", "png", "jpeg", "tif"])
    chapter = st.sidebar.selectbox("📘 Chọn chương", list(CHAPTER_OPS.keys()))
    op = st.sidebar.selectbox("⚙️ Chọn phép biến đổi", list(CHAPTER_OPS[chapter].keys()))
    func = CHAPTER_OPS[chapter][op]

    # Lấy ảnh đầu vào
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
    else:
        default_path = DEFAULT_IMAGES.get(chapter, {}).get(op)
        if default_path:
            img = Image.open(default_path).convert("RGB")
        else:
            return

    # Chuyển ảnh sang OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Thực hiện xử lý
    if func in [c3.Negative_Color, c3.HistEqualColor, c9.Contour]:
        out = func(img_cv)
    else:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        out_gray = func(gray)
        out = cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)

    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Kết quả")
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
