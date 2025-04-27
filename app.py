import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Định nghĩa tên các lớp biển báo
sign_names = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

MODEL_PATH = 'models/best_gtsrb_model.keras'
IMG_HEIGHT = 32
IMG_WIDTH = 32

@st.cache_resource
def load_gtsrb_model():
    model = load_model(MODEL_PATH)
    return model

model = load_gtsrb_model()

st.title("Phân loại biển báo giao thông GTSRB")
st.write("Tải lên một ảnh biển báo giao thông (ảnh đã cắt focus vào biển báo) để nhận dự đoán.")

uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png", "ppm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ảnh bạn đã tải lên', use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = int(np.argmax(predictions[0]))
    predicted_prob = float(np.max(predictions[0]))

    st.markdown(f"### Dự đoán: {sign_names[predicted_class]}")
    st.write(f"Xác suất dự đoán: {predicted_prob:.2%}")

    # Hiển thị top 5 dự đoán
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    st.write("#### Top 5 dự đoán:")
    for idx in top_5_indices:
        st.write(f"**Lớp {idx}**: {sign_names[idx]}  \nXác suất: {predictions[0][idx]:.2%}")

else:
    st.info("Vui lòng tải lên một ảnh để nhận dự đoán.") 