from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

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

# Khởi tạo FastAPI
app = FastAPI(
    title="GTSRB Traffic Sign Classification API",
    description="API nhận diện biển báo giao thông Đức (GTSRB) sử dụng mô hình TensorFlow/Keras.",
    version="1.0"
)

# Cho phép CORS nếu cần (giúp frontend truy cập API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model khi khởi động server
model = load_model(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh từ request
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img = np.array(image)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Dự đoán
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        predicted_prob = float(np.max(predictions[0]))

        # Top 5 dự đoán
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5 = [
            {
                "class_id": int(idx),
                "class_name": sign_names[idx],
                "probability": float(predictions[0][idx])
            }
            for idx in top_5_indices
        ]

        result = {
            "predicted_class_id": predicted_class,
            "predicted_class_name": sign_names[predicted_class],
            "predicted_probability": predicted_prob,
            "top_5": top_5
        }
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)