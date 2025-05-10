import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

def get_file_path(filename):
    static_path = os.path.join("static", filename)
    return static_path if os.path.exists(static_path) else filename

def standardize_image(image, size=(416, 416)):
    return cv2.resize(image, size)

def app():
    # Constants
    inpWidth, inpHeight = 416, 416
    confThreshold, nmsThreshold = 0.5, 0.4

    # Load mô hình và lớp
    model_path = get_file_path("garbage.onnx")
    filename_classes = get_file_path("garbage.txt")

    if "GarbageNet" not in st.session_state:
        st.session_state["GarbageNet"] = cv2.dnn.readNet(model_path)

    with open(filename_classes, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")

    selected_class_ids = list(range(len(classes)))

    def process_output(out, box_scale, selected_class_ids):
        detections = []
        for detection in out.transpose(1, 0):
            scores = detection[4:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold and classId in selected_class_ids:
                center_x, center_y, width, height = detection[:4]
                left = int((center_x - width / 2) * box_scale[0])
                top = int((center_y - height / 2) * box_scale[1])
                width = int(width * box_scale[0])
                height = int(height * box_scale[1])
                detections.append((classId, confidence, [left, top, width, height]))
        return detections

    def postprocess(frame, outs, selected_class_ids):
        frameHeight, frameWidth = frame.shape[:2]
        box_scale = (frameWidth / inpWidth, frameHeight / inpHeight)

        detections = []
        for out in outs:
            detections.extend(process_output(out[0], box_scale, selected_class_ids))

        if detections:
            classIds, confidences, boxes = zip(*detections)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

            for i in indices.flatten():
                box = boxes[i]
                left, top, width, height = box
                drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

    group_color_map = {
        "BIODEGRADABLE": (0, 140, 255),  # cam - hữu cơ
        "PAPER": (0, 255, 0),            # xanh - tái chế
        "CARDBOARD": (0, 255, 0),
        "PLASTIC": (0, 255, 0),
        "METAL": (0, 255, 0),
        "GLASS": (0, 255, 0),
    }

    def drawPred(frame, classId, conf, left, top, right, bottom):
        class_name = classes[classId].strip().upper()
        box_color = group_color_map.get(class_name, (200, 200, 200))  # Mặc định màu xám nếu không có

        # Vẽ bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness=2)

        # Chuẩn bị label
        label = f"{class_name}: {conf:.2f}"
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])

        # Vẽ nền cho label (màu trắng)
        cv2.rectangle(
            frame,
            (left, top - labelSize[1]),
            (left + labelSize[0], top + baseLine),
            (255, 255, 255),
            cv2.FILLED,
        )

        # Vẽ chữ (màu đen)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



    # UI
    st.title("🗑️ Phân loại rác thải")

    img_file_buffer = st.file_uploader("📂 Chọn ảnh rác cần phân loại", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"])
    cols = st.columns(2)

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = standardize_image(frame, (inpWidth, inpHeight))

        with cols[0]:
            st.subheader("🖼️ Ảnh gốc")
            st.image(frame, channels="BGR", use_container_width=True)

        with cols[1]:
            text_container = st.empty()
            img_container = st.empty()

        if st.button("🔍 Dự đoán"):
            blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)
            st.session_state["GarbageNet"].setInput(blob, scalefactor=0.00392)
            outs = st.session_state["GarbageNet"].forward(st.session_state["GarbageNet"].getUnconnectedOutLayersNames())
            postprocess(frame, outs, selected_class_ids)
            text_container.subheader("📌 Kết quả nhận dạng")
            img_container.image(frame, channels="BGR", use_container_width=True)