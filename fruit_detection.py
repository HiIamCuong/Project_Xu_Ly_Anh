import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os

def get_file_path(filename):
    static_path = os.path.join("static", filename)
    return static_path if os.path.exists(static_path) else filename

def standardize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

def app():
    # Constants
    inpWidth, inpHeight = 640, 640
    confThreshold, nmsThreshold = 0.5, 0.4

    # Load mô hình và lớp
    model_path = get_file_path("trai_cay.onnx")
    filename_classes = get_file_path("trai_cay.txt")

    if "Net" not in st.session_state:
        st.session_state["Net"] = cv2.dnn.readNet(model_path)

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

    def drawPred(frame, classId, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label = f"{classes[classId]}: {conf:.2f}"
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(
            frame,
            (left, top - labelSize[1]),
            (left + labelSize[0], top + baseLine),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # UI
    st.title("🍎 Nhận dạng trái cây với mô hình YOLOv8 (ONNX)")
    st.write("Hệ thống nhận dạng trái cây dựa trên mô hình `trai_cay.onnx`.")

    img_file_buffer = st.file_uploader("📂 Chọn ảnh", type=["bmp", "png", "jpg", "jpeg", "tif", "gif"])
    cols = st.columns(2)

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = standardize_image(frame, (inpWidth, inpHeight))

        with cols[0]:
            st.subheader("Ảnh gốc")
            st.image(frame, channels="BGR")

        with cols[1]:
            text_container = st.empty()
            img_container = st.empty()

        if st.button("🔍 Dự đoán"):
            blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)
            st.session_state["Net"].setInput(blob, scalefactor=0.00392)
            outs = st.session_state["Net"].forward(st.session_state["Net"].getUnconnectedOutLayersNames())
            postprocess(frame, outs, selected_class_ids)
            text_container.subheader("📌 Kết quả")
            img_container.image(frame, channels="BGR")