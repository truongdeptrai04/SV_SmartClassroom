from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
from datetime import datetime
from services.face_recognition_service import FaceRecognitionService
from services.emotion_detection_service import EmotionDetectionService

app = Flask(__name__)
face_service = FaceRecognitionService()
emotion_service = EmotionDetectionService()

IMAGE_DIR = "received_images"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'])
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"{IMAGE_DIR}/received_{timestamp}.jpg"
            cv2.imwrite(img_filename, image)
            print(f"Đã lưu ảnh tại: {img_filename}")
        else:
            print("Không thể giải mã ảnh từ base64")

        face_results = emotion_service.detect_faces_and_emotions(image)
        if not face_results:
            print("Không phát hiện khuôn mặt nào trong ảnh")
            return jsonify({"results": [], "message": "No faces detected"}), 200

        print(f"Số kết quả nhận diện: {len(face_results)}")  # Debug
        results = []
        for result in face_results:
            face_roi = result["face"]
            student_name = face_service.recognize(face_roi)
            emotion_percentages = result["emotion_percentages"]
            results.append({
                "student_name": student_name,
                "emotion_percentages": emotion_percentages
            })

        return jsonify({"results": results})
    except Exception as e:
        print(f"Lỗi: {str(e)}")  # Debug
        return jsonify({"error": str(e)}), 500

@app.route('/add_student', methods=['POST'])
def add_student():
    try:
        data = request.get_json()
        name = data['name']
        image_list = data['images']

        images = []
        for img_data in image_list:
            decoded = base64.b64decode(img_data)
            image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)

        if not images:
            return jsonify({"error": "No valid images provided"}), 400

        if not face_service.add_face(name, images):
            return jsonify({"error": "Failed to generate face encodings"}), 400

        return jsonify({"message": f"Added {name} successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Smart Classroom API"})

if __name__ == "__main__":
    from pyngrok import ngrok
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    ngrok.set_auth_token("2u0rqcmZCraUTiXy8NtAyk9wVhT_6YNty8g1YBdcEwDM9FUgV")
    public_url = ngrok.connect(5001)
    print(f"Server running at: {public_url}")
    app.run(host="0.0.0.0", port=5001)