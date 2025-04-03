from datetime import datetime

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from services.face_recognition_service import FaceRecognitionService
from services.emotion_detection_service import EmotionDetectionService
from services.firestore_service import FirestoreService

app = Flask(__name__)
face_service = FaceRecognitionService()
emotion_service = EmotionDetectionService()
firestore_service = FirestoreService()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        faces = data.get('faces', [])

        results = []
        for face_data in faces:
            img_data = base64.b64decode(face_data)
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            student_name = face_service.recognize(image)
            if student_name == "Unknown":
                results.append({"student_name": student_name, "error": "Unknown Student"})
                continue

            emotion_percentages = emotion_service.detect(image)
            if not emotion_percentages:
                results.append({"student_name": student_name, "error": "No Emotion detected"})
                continue

            student_id, class_id = firestore_service.get_student_by_name(student_name)
            if not student_id:
                results.append({"student_name": student_name, "error": "Student not in firebase"})
                continue
            firestore_service.save_student_emotion(student_id, class_id, emotion_percentages)
            results.append({"student_name": student_name, "emotion_percentages":emotion_percentages})

        return jsonify("result", results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/end_class', methods=['POST'])
def end_class():
    try:
        data = request.get_json()
        class_id = data.get('classId')
        start_time = datetime.fromisoformat(data['startTime'])
        end_time = datetime.fromisoformat(data['endTime'])

        firestore_service.update_class_emotion_stats(class_id, start_time, end_time)
        return jsonify({"message" : f"class {class_id} emotion stats updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_student', method = ['POST'])
def add_student():
    try:
        data = request.get_json()
        name = data['name']
        class_id = data['classId']
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

        student_data = {
            'classId': class_id,
            'dateOfBirth': data.get('dateOfBirth', ''),
            'email': data.get('email', ''),
            'gender': data.get('gender', ''),
            'phone': data.get('phone', ''),
            'studentCode': data.get('studentCode', ''),
            'studentName': name
        }
        student_id = firestore_service.add_student(student_data)

        return jsonify({"message": f"Added {name} successfully", "studentId": student_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print(f"Server running at: {public_url}")
    app.run(host='0.0.0.0', port=5000)
