from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import pytz
from pyngrok import ngrok
from services.face_recognition_service import FaceRecognitionService
from services.emotion_detection_service import EmotionDetectionService
from services.firestore_service import FirestoreService
import time
from google.cloud.firestore_v1.base_query import FieldFilter
from firebase_admin import firestore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

app = Flask(__name__)
face_service = FaceRecognitionService()
emotion_service = EmotionDetectionService()
firestore_service = FirestoreService()
scheduler = BackgroundScheduler()
VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')  # UTC+7

IMAGE_DIR = "received_images"
last_processed = {}  # Lưu thời gian xử lý cuối cùng cho mỗi classId

def schedule_emotion_summary(class_id, start_time, end_time, day_of_week):
    # Chuyển thời gian sang UTC+7
    end_time_vn = end_time.astimezone(VN_TIMEZONE)
    # Chuyển day_of_week sang định dạng APScheduler (mon, tue, ...)
    day_map = {
        'Monday': 'mon', 'Tuesday': 'tue', 'Wednesday': 'wed', 'Thursday': 'thu',
        'Friday': 'fri', 'Saturday': 'sat', 'Sunday': 'sun'
    }
    cron_day = day_map.get(day_of_week, 'sat')  # Mặc định là Saturday nếu không khớp
    # Lên lịch lặp lại hàng tuần tại giờ, phút và ngày trong tuần
    print(f"Scheduling weekly emotion summary for class {class_id} at {end_time_vn.strftime('%H:%M')} on {day_of_week} (VN)")
    scheduler.add_job(
        func=firestore_service.update_class_emotion_stats,
        trigger=CronTrigger(day_of_week=cron_day, hour=end_time_vn.hour, minute=end_time_vn.minute, timezone=VN_TIMEZONE),
        args=[class_id, start_time, end_time],
        id=f"summary_{class_id}",
        replace_existing=True
    )

def schedule_all_classes():
    # Lấy tất cả lớp học từ Firestore
    docs = firestore_service.db.collection('Classes').stream()
    for doc in docs:
        class_data = doc.to_dict()
        start_time = class_data['startTime'].astimezone(VN_TIMEZONE)
        end_time = class_data['endTime'].astimezone(VN_TIMEZONE)
        day_of_week = class_data['dayOfWeek']
        schedule_emotion_summary(doc.id, start_time, end_time, day_of_week)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Lấy thời gian hiện tại (UTC+7)
        current_time = datetime.now(VN_TIMEZONE)
        current_day = current_time.strftime('%A')
        print(f"Processing image at {current_time} on {current_day}")

        # Tìm lớp học đang diễn ra
        class_id, class_data = firestore_service.get_class_by_time(current_time, current_day)
        if not class_id:
            return jsonify({"error": "No class in session at this time"}), 400

        start_time = class_data['startTime'].astimezone(VN_TIMEZONE)
        end_time = class_data['endTime'].astimezone(VN_TIMEZONE)
        print(f"Found class {class_id}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} (VN)")

        # Kiểm tra tần suất xử lý (15 giây/lần cho mỗi classId)
        if class_id in last_processed and time.time() - last_processed[class_id] < 15:
            return jsonify({"error": "Processing too frequent, please wait"}), 429
        last_processed[class_id] = time.time()

        # Giải mã hình ảnh
        img_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Lưu ảnh (tùy chọn, để debug)
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{IMAGE_DIR}/received_{timestamp_str}.jpg"
        cv2.imwrite(img_filename, image)
        print(f"Đã lưu ảnh tại: {img_filename}")

        # Nhận diện khuôn mặt và cảm xúc
        face_results = emotion_service.detect_faces_and_emotions(image)
        if not face_results:
            print("Không phát hiện khuôn mặt nào trong ảnh")
            return jsonify({"results": [], "message": "No faces detected"}), 200

        results = []
        for result in face_results:
            face_roi = result["face"]
            student_name = face_service.recognize(face_roi)
            if student_name == "Unknown":
                continue  # Bỏ qua nếu không nhận diện được học sinh

            # Lấy student_id và class_id từ Firestore
            student_id, _ = firestore_service.get_student_by_name(student_name)
            if not student_id:
                continue  # Bỏ qua nếu không tìm thấy học sinh

            emotion_percentages = result["emotion_percentages"]
            # Lưu vào StudentEmotionStats
            firestore_service.save_student_emotion(student_id, class_id, emotion_percentages)

            # Kiểm tra cảm xúc tiêu cực (sad > 50% hoặc angry > 50% trong 3 lần liên tiếp)
            recent_stats = firestore_service.db.collection('StudentEmotionStats')\
                .where(filter=FieldFilter('studentId', '==', student_id))\
                .where(filter=FieldFilter('classId', '==', class_id))\
                .order_by('createAt', direction=firestore.Query.DESCENDING)\
                .limit(3).stream()
            sad_values = [stat.to_dict()['sad'] for stat in recent_stats]
            recent_stats = firestore_service.db.collection('StudentEmotionStats')\
                .where(filter=FieldFilter('studentId', '==', student_id))\
                .where(filter=FieldFilter('classId', '==', class_id))\
                .order_by('createAt', direction=firestore.Query.DESCENDING)\
                .limit(3).stream()
            angry_values = [stat.to_dict()['angry'] for stat in recent_stats]

            if len(sad_values) >= 3 and all(s > 50 for s in sad_values) or \
               len(angry_values) >= 3 and all(a > 50 for a in angry_values):
                firestore_service.db.collection('Alerts').add({
                    'alertId': f"alert_{student_id}_{timestamp_str}",
                    'classId': class_id,
                    'studentId': student_id,
                    'title': "Cảnh báo cảm xúc tiêu cực",
                    'content': f"Học sinh {student_name} có cảm xúc tiêu cực (sad: {emotion_percentages['sad']}%, angry: {emotion_percentages['angry']}%)",
                    'timestamp': firestore.SERVER_TIMESTAMP
                })

            results.append({
                "student_name": student_name,
                "student_id": student_id,
                "emotion_percentages": emotion_percentages
            })

        return jsonify({"results": results, "message": "Processed successfully"})
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/summarize_class_emotions', methods=['POST'])
def summarize_class_emotions():
    try:
        data = request.get_json()
        class_id = data.get('classId')
        start_time = data.get('startTime')  # Timestamp từ client
        end_time = data.get('endTime')      # Timestamp từ client

        # Chuyển đổi timestamp nếu cần
        start_time = datetime.fromisoformat(start_time.replace('Z', '+07:00')).astimezone(VN_TIMEZONE)
        end_time = datetime.fromisoformat(end_time.replace('Z', '+07:00')).astimezone(VN_TIMEZONE)

        firestore_service.update_class_emotion_stats(class_id, start_time, end_time)
        return jsonify({"message": "Class emotion stats updated successfully"})
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_class_schedule', methods=['GET'])
def check_class_schedule():
    try:
        class_id = request.args.get('classId')
        print(f"Checking schedule for classId: {class_id}")
        class_ref = firestore_service.db.collection('Classes').document(class_id).get()
        if not class_ref.exists:
            print(f"Class {class_id} not found in Firestore")
            return jsonify({"error": "Class not found"}), 404

        class_data = class_ref.to_dict()
        print(f"Found class {class_id}: {class_data}")
        return jsonify({
            "classId": class_id,
            "startTime": class_data['startTime'].astimezone(VN_TIMEZONE).strftime('%H:%M'),
            "endTime": class_data['endTime'].astimezone(VN_TIMEZONE).strftime('%H:%M'),
            "dayOfWeek": class_data['dayOfWeek']
        })
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'])
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            timestamp = datetime.now(VN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
            img_filename = f"{IMAGE_DIR}/received_{timestamp}.jpg"
            cv2.imwrite(img_filename, image)
            print(f"Đã lưu ảnh tại: {img_filename}")
        else:
            print("Không thể giải mã ảnh từ base64")

        face_results = emotion_service.detect_faces_and_emotions(image)
        if not face_results:
            print("Không phát hiện khuôn mặt nào trong ảnh")
            return jsonify({"results": [], "message": "No faces detected"}), 200

        print(f"Số kết quả nhận diện: {len(face_results)}")
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
        print(f"Lỗi: {str(e)}")
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

        # Thêm học sinh vào Firestore
        student_data = {
            "studentName": name,
            "classId": data.get('classId', ''),
            "email": data.get('email', ''),
            "gender": data.get('gender', ''),
            "phone": data.get('phone', ''),
            "status": "active",
            "studentCode": data.get('studentCode', ''),
            "userId": data.get('userId', '')
        }
        student_id = firestore_service.add_student(student_data)

        return jsonify({"message": f"Added {name} successfully", "studentId": student_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to Smart Classroom API"})

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    scheduler.start()  # Khởi động scheduler
    schedule_all_classes()  # Lên lịch tổng hợp cho tất cả lớp học
    ngrok.set_auth_token("2u0rqcmZCraUTiXy8NtAyk9wVhT_6YNty8g1YBdcEwDM9FUgV")
    public_url = ngrok.connect(5001, bind_tls=True).public_url
    print(f"Server running at: {public_url}")
    app.run(host="0.0.0.0", port=5001)