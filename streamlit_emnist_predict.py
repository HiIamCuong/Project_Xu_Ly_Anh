import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2
import random

def app():
    st.title("🧠 Nhận diện ký tự viết tay (EMNIST)")

    emnist_labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def tao_anh_ngau_nhien():
        image = np.zeros((10 * 28, 10 * 28), np.uint8)
        data = np.zeros((100, 28, 28, 1), np.uint8)

        for i in range(100):
            n = random.randint(0, len(st.session_state.X_test) - 1)
            sample = st.session_state.X_test[n]
            data[i] = st.session_state.X_test[n]
            x = i // 10
            y = i % 10
            image[x*28:(x+1)*28, y*28:(y+1)*28] = sample[:, :, 0]
        return image, data

    # Load model và dữ liệu EMNIST
    if 'is_load' not in st.session_state:
        st.session_state.model = tf.keras.models.load_model("char_recognition_model.keras")

        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/byclass', split=['train', 'test'],
            shuffle_files=True, as_supervised=True, with_info=True
        )

        X_test = []
        for img, _ in tfds.as_numpy(ds_test.take(10000)):
            X_test.append(img)
        X_test = np.array(X_test).reshape((-1, 28, 28, 1))

        st.session_state.X_test = X_test
        st.session_state.is_load = True

    # Layout gồm 2 cột: ảnh và kết quả
    col1, col2 = st.columns([1, 1.2])

    with col1:
        if st.button("📤 Tạo ảnh"):
            image, data = tao_anh_ngau_nhien()
            st.session_state.image = image
            st.session_state.data = data

        if 'image' in st.session_state:
            st.image(st.session_state.image, clamp=True, width=280)

    with col2:
        if 'image' in st.session_state:
            if st.button("🤖 Nhận dạng"):
                data = st.session_state.data.astype('float32') / 255.0
                ket_qua = st.session_state.model.predict(data)
                s = ''
                for i, x in enumerate(ket_qua):
                    s += emnist_labels[np.argmax(x)] + ' '
                    if (i + 1) % 10 == 0:
                        s += '\n'
                st.session_state.result = s

            if 'result' in st.session_state:
                styled_result = "<div style='font-family: monospace; font-size: 16px; line-height: 28px; white-space: pre;'>"
                styled_result += st.session_state.result
                styled_result += "</div>"
                st.markdown(styled_result, unsafe_allow_html=True)
