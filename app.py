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
import logging

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    handlers=[
        logging.FileHandler('smartclassroom.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ProcessImage')

face_service = FaceRecognitionService()
emotion_service = EmotionDetectionService()
firestore_service = FirestoreService()
scheduler = BackgroundScheduler()
VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')

IMAGE_DIR = "received_images"
last_processed = {}

# Danh sách cảm xúc mong đợi
EXPECTED_EMOTIONS = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def schedule_emotion_summary(class_id, start_time, end_time, day_of_week):
    try:
        start_time_vn = datetime(
            start_time.year,
            start_time.month,
            start_time.day,
            start_time.hour,
            start_time.minute,
            start_time.second,
            start_time.microsecond,
            tzinfo=start_time.tzinfo
        ).astimezone(VN_TIMEZONE)
        end_time_vn = datetime(
            end_time.year,
            end_time.month,
            end_time.day,
            end_time.hour,
            end_time.minute,
            end_time.second,
            end_time.microsecond,
            tzinfo=end_time.tzinfo
        ).astimezone(VN_TIMEZONE)
        day_map = {
            'Monday': 'mon', 'Tuesday': 'tue', 'Wednesday': 'wed', 'Thursday': 'thu',
            'Friday': 'fri', 'Saturday': 'sat', 'Sunday': 'sun'
        }
        cron_day = day_map.get(day_of_week, 'sat')
        logger.info(f"Scheduling emotion summary for class {class_id} at {end_time_vn.strftime('%H:%M')} on {day_of_week} (VN), start_time={start_time_vn}, end_time={end_time_vn}")
        scheduler.add_job(
            func=firestore_service.update_class_emotion_stats,
            trigger=CronTrigger(day_of_week=cron_day, hour=end_time_vn.hour, minute=end_time_vn.minute, timezone=VN_TIMEZONE),
            args=[class_id, start_time_vn, end_time_vn],
            id=f"summary_{class_id}",
            replace_existing=True
        )
        logger.info(f"Scheduled job summary_{class_id}, next run: {scheduler.get_job(f'summary_{class_id}').next_run_time}")
    except Exception as e:
        logger.error(f"Error scheduling emotion summary for class {class_id}: {str(e)}")

def schedule_all_classes():
    try:
        logger.info("Refreshing scheduler for all classes")
        scheduler.remove_all_jobs()  # Xóa các job cũ
        docs = firestore_service.db.collection('Classes').stream()
        for doc in docs:
            class_data = doc.to_dict()
            start_time = class_data['startTime']
            end_time = class_data['endTime']
            day_of_week = class_data['dayOfWeek']
            schedule_emotion_summary(doc.id, start_time, end_time, day_of_week)
        logger.info("Completed scheduling for all classes")
    except Exception as e:
        logger.error(f"Error scheduling classes: {str(e)}")

def setup_firestore_listener():
    def on_snapshot(col_snapshot, changes, read_time):
        logger.info("Detected changes in Classes collection")
        schedule_all_classes()
    firestore_service.db.collection('Classes').on_snapshot(on_snapshot)
    logger.info("Firestore listener for Classes collection initialized")

@app.route('/refresh_schedule', methods=['POST'])
def refresh_schedule():
    try:
        schedule_all_classes()
        logger.info("[REFRESH_SCHEDULE]: Scheduler refreshed successfully")
        return jsonify({"message": "Scheduler refreshed successfully"}), 200
    except Exception as e:
        logger.error(f"[REFRESH_SCHEDULE]: Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        start_request = time.time()
        current_time = datetime.now(VN_TIMEZONE)
        current_day = current_time.strftime('%A')
        logger.info(f"[START_REQUEST]: Processing image at {current_time} on {current_day}")

        # Nhận và kiểm tra dữ liệu
        data = request.get_json()
        image_base64 = data.get('image')
        if not image_base64:
            logger.error("[DECODE_IMAGE]: No image provided")
            return jsonify({"error": "No image provided"}), 400

        # Tìm lớp học
        class_id, class_data = firestore_service.get_class_by_time(current_time, current_day)
        if not class_id:
            logger.error("[CHECK_CLASS]: No class in session at this time")
            return jsonify({"error": "No class in session at this time"}), 400

        start_time = datetime(
            class_data['startTime'].year,
            class_data['startTime'].month,
            class_data['startTime'].day,
            class_data['startTime'].hour,
            class_data['startTime'].minute,
            class_data['startTime'].second,
            class_data['startTime'].microsecond,
            tzinfo=class_data['startTime'].tzinfo
        ).astimezone(VN_TIMEZONE)
        end_time = datetime(
            class_data['endTime'].year,
            class_data['endTime'].month,
            class_data['endTime'].day,
            class_data['endTime'].hour,
            class_data['endTime'].minute,
            class_data['endTime'].second,
            class_data['endTime'].microsecond,
            tzinfo=class_data['endTime'].tzinfo
        ).astimezone(VN_TIMEZONE)
        logger.info(f"[CHECK_CLASS]: Found class {class_id}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} (VN)")

        # Kiểm tra và cập nhật scheduler nếu cần
        job = scheduler.get_job(f"summary_{class_id}")
        if not job or job.next_run_time.tzinfo != VN_TIMEZONE or \
           job.next_run_time.strftime('%H:%M') != end_time.strftime('%H:%M') or \
           job.args != [class_id, start_time, end_time]:
            logger.warning(f"[SCHEDULER]: Updating scheduler for class {class_id}")
            schedule_emotion_summary(class_id, class_data['startTime'], class_data['endTime'], class_data['dayOfWeek'])

        # Kiểm tra tần suất xử lý (15s)
        if class_id in last_processed and time.time() - last_processed[class_id] < 15:
            logger.warning(f"[RATE_LIMIT]: Processing too frequent for class {class_id}, last processed {time.time() - last_processed[class_id]:.2f}s ago")
            return jsonify({"error": "Processing too frequent, please wait", "retry_after": 15}), 429, {"Retry-After": "15"}
        last_processed[class_id] = time.time()

        # Giải mã ảnh
        start_decode = time.time()
        img_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("[DECODE_IMAGE]: Invalid image data")
            return jsonify({"error": "Invalid image data"}), 400
        decode_time = time.time() - start_decode
        logger.info(f"[DECODE_IMAGE]: Decoded base64 image, shape={image.shape} (took {decode_time:.2f}s)")

        # Nhận diện khuôn mặt và cảm xúc
        start_detect = time.time()
        face_results = emotion_service.detect_faces_and_emotions(image)
        detect_time = time.time() - start_detect
        if not face_results:
            logger.info(f"[DETECT_FACES_EMOTIONS]: No faces detected (took {detect_time:.2f}s)")
            # Lưu ảnh gốc nếu không có khuôn mặt
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            img_filename = f"{IMAGE_DIR}/received_{timestamp_str}.jpg"
            cv2.imwrite(img_filename, image)
            logger.info(f"[SAVE_IMAGE]: Saved image at {img_filename}")
            return jsonify({"results": [], "message": "No faces detected"}), 200
        logger.info(f"[DETECT_FACES_EMOTIONS]: Detected {len(face_results)} faces and emotions (took {detect_time:.2f}s)")

        # Lấy tọa độ khuôn mặt từ EmotionDetectionService
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = emotion_service.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # Vẽ khung và cảm xúc lên ảnh
        for idx, ((x, y, w, h), result) in enumerate(zip(faces, face_results)):
            emotion_percentages = result["emotion_percentages"]
            # Tìm cảm xúc lớn nhất
            dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
            dominant_percentage = emotion_percentages[dominant_emotion]
            # Vẽ khung màu xanh quanh khuôn mặt
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Thêm văn bản cảm xúc (ví dụ: "Neutral - 62%")
            text = f"{dominant_emotion} - {dominant_percentage:.0f}%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Lưu ảnh đã vẽ
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
        img_filename = f"{IMAGE_DIR}/received_{timestamp_str}.jpg"
        cv2.imwrite(img_filename, image)
        logger.info(f"[SAVE_IMAGE]: Saved image with face boxes and emotions at {img_filename}")

        # Nhận diện học sinh và lưu Firestore
        results = []
        start_recognize = time.time()
        recognize_times = []
        for result in face_results:
            face_roi = result["face"]
            emotion_percentages = result["emotion_percentages"]

            # Kiểm tra emotion_percentages
            if set(emotion_percentages.keys()) != set(EXPECTED_EMOTIONS):
                logger.error(f"[DETECT_FACES_EMOTIONS]: Invalid emotion keys: {emotion_percentages.keys()}")
                return jsonify({"error": "Invalid emotion percentages"}), 500
            logger.info(f"[DETECT_FACES_EMOTIONS]: Emotion percentages: {emotion_percentages}")

            # Nhận diện học sinh
            recognize_start = time.time()
            student_name, student_id = face_service.recognize(face_roi, class_id)
            recognize_time = time.time() - recognize_start
            recognize_times.append(recognize_time)

            # Kiểm tra sinh viên thuộc lớp qua StudentClasses
            if student_id and student_id != "Unknown":
                student_class = firestore_service.db.collection('StudentClasses')\
                    .where(filter=FieldFilter('studentId', '==', student_id))\
                    .where(filter=FieldFilter('classId', '==', class_id))\
                    .limit(1).stream()
                if not any(student_class):
                    logger.warning(f"[RECOGNIZE_STUDENTS]: Student {student_id} not in class {class_id}")
                    student_name = "Unknown"
                    student_id = None

            if student_name == "Unknown":
                logger.warning(f"[RECOGNIZE_STUDENTS]: Unknown student for face in class {class_id}")

            # Lưu StudentEmotionStats cho sinh viên được nhận diện
            if student_id and student_id != "Unknown":
                firestore_service.save_student_emotion(student_id, class_id, emotion_percentages)

                # Kiểm tra cảm xúc tiêu cực
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
                        'content': f"Học sinh {student_name} có ({emotion_percentages['Sad']}) ({emotion_percentages['Angry']})",
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })

            # Thêm vào kết quả (bao gồm cả Unknown)
            results.append({
                "student_name": student_name,
                "student_id": student_id,
                "emotion_percentages": emotion_percentages
            })

        recognize_total = time.time() - start_recognize
        avg_recognize_time = sum(recognize_times) / len(recognize_times) if recognize_times else 0
        logger.info(f"[RECOGNIZE_STUDENTS]: Recognized {len([r for r in results if r['student_id']]) for r in results} students "
                    f"(avg {avg_recognize_time:.3f}s/face, total {recognize_total:.2f}s)")
        logger.info(f"[FIRESTORE_OPERATIONS]: Saved emotion stats and checked alerts for "
                    f"{len([r for r in results if r['student_id']]) for r in results} students (took {recognize_total:.2f}s)")

        total_time = time.time() - start_request
        logger.info(f"[END_REQUEST]: Completed request for class_id={class_id} (total {total_time:.2f}s)")

        return jsonify({"results": results, "message": "Processed successfully"}), 200

    except Exception as e:
        logger.error(f"[ERROR]: Failed to process request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_request = time.time()
        data = request.get_json()
        image_base64 = data.get('image')
        class_id = data.get('classId')
        if not image_base64:
            logger.error("[DECODE_IMAGE]: No image provided")
            return jsonify({"error": "No image provided"}), 400
        if not class_id:
            logger.error("[CHECK_CLASS]: No classId provided")
            return jsonify({"error": "No classId provided"}), 400

        # Giải mã ảnh
        start_decode = time.time()
        img_data = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.error("[DECODE_IMAGE]: Invalid image data")
            return jsonify({"error": "Invalid image data"}), 400
        decode_time = time.time() - start_decode
        logger.info(f"[DECODE_IMAGE]: Decoded base64 image, shape={image.shape} (took {decode_time:.2f}s)")

        # Lưu ảnh để debug
        timestamp = datetime.now(VN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        img_filename = f"{IMAGE_DIR}/received_{timestamp}.jpg"
        cv2.imwrite(img_filename, image)
        logger.info(f"[SAVE_IMAGE]: Saved image at {img_filename}")

        # Nhận diện khuôn mặt và cảm xúc
        start_detect = time.time()
        face_results = emotion_service.detect_faces_and_emotions(image)
        detect_time = time.time() - start_detect
        if not face_results:
            logger.info(f"[DETECT_FACES_EMOTIONS]: No faces detected (took {detect_time:.2f}s)")
            return jsonify({"results": [], "message": "No faces detected"}), 200
        logger.info(f"[DETECT_FACES_EMOTIONS]: Detected {len(face_results)} faces (took {detect_time:.2f}s)")

        # Nhận diện học sinh
        start_recognize = time.time()
        results = []
        recognize_times = []
        for result in face_results:
            face_roi = result["face"]
            emotion_percentages = result["emotion_percentages"]

            # Kiểm tra emotion_percentages
            if set(emotion_percentages.keys()) != set(EXPECTED_EMOTIONS):
                logger.error(f"[DETECT_FACES_EMOTIONS]: Invalid emotion keys: {emotion_percentages.keys()}")
                return jsonify({"error": "Invalid emotion percentages"}), 500
            logger.info(f"[DETECT_FACES_EMOTIONS]: Emotion percentages: {emotion_percentages}")

            recognize_start = time.time()
            student_name, student_id = face_service.recognize(face_roi, class_id)
            recognize_time = time.time() - recognize_start
            recognize_times.append(recognize_time)
            results.append({
                "student_name": student_name,
                "student_id": student_id,
                "emotion_percentages": emotion_percentages
            })

        recognize_total_time = time.time() - start_recognize
        avg_recognize_time = sum(recognize_times) / len(recognize_times) if recognize_times else 0
        logger.info(f"[RECOGNIZE_STUDENTS]: Recognized {len([r for r in results if r['student_id']])} students "
                    f"(avg {avg_recognize_time:.3f}s/face, total {recognize_total_time:.2f}s)")

        total_time = time.time() - start_request
        logger.info(f"[END_REQUEST]: Completed predict request (total {total_time:.2f}s)")

        return jsonify({"results": results, "message": "Processed successfully"}), 200

    except Exception as e:
        logger.error(f"[ERROR]: Failed to process predict request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/summarize_class_emotions', methods=['POST'])
def summarize_class_emotions():
    try:
        data = request.get_json()
        class_id = data.get('classId')
        start_time = data.get('startTime')
        end_time = data.get('endTime')

        start_time = datetime.fromisoformat(start_time.replace('Z', '+07:00')).astimezone(VN_TIMEZONE)
        end_time = datetime.fromisoformat(end_time.replace('Z', '+07:00')).astimezone(VN_TIMEZONE)

        firestore_service.update_class_emotion_stats(class_id, start_time, end_time)
        logger.info(f"[SUMMARIZE_EMOTIONS]: Updated emotion stats for class {class_id}")
        return jsonify({"message": "Class emotion stats updated successfully"}), 200
    except Exception as e:
        logger.error(f"[SUMMARIZE_EMOTIONS]: Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_class_schedule', methods=['GET'])
def check_class_schedule():
    try:
        class_id = request.args.get('classId')
        logger.info(f"[CHECK_SCHEDULE]: Checking schedule for classId={class_id}")
        class_ref = firestore_service.db.collection('Classes').document(class_id).get()
        if not class_ref.exists:
            logger.error(f"[CHECK_SCHEDULE]: Class {class_id} not found in Firestore")
            return jsonify({"error": "Class not found"}), 404

        class_data = class_ref.to_dict()
        start_time = datetime(
            class_data['startTime'].year,
            class_data['startTime'].month,
            class_data['startTime'].day,
            class_data['startTime'].hour,
            class_data['startTime'].minute,
            class_data['startTime'].second,
            class_data['startTime'].microsecond,
            tzinfo=class_data['startTime'].tzinfo
        ).astimezone(VN_TIMEZONE)
        end_time = datetime(
            class_data['endTime'].year,
            class_data['endTime'].month,
            class_data['endTime'].day,
            class_data['endTime'].hour,
            class_data['endTime'].minute,
            class_data['endTime'].second,
            class_data['endTime'].microsecond,
            tzinfo=class_data['endTime'].tzinfo
        ).astimezone(VN_TIMEZONE)
        logger.info(f"[CHECK_SCHEDULE]: Found class {class_id}: {class_data}")
        return jsonify({
            "classId": class_id,
            "startTime": start_time.strftime('%H:%M'),
            "endTime": end_time.strftime('%H:%M'),
            "dayOfWeek": class_data['dayOfWeek']
        }), 200
    except Exception as e:
        logger.error(f"[CHECK_SCHEDULE]: Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_student_with_images', methods=['POST'])
def add_student_with_images():
    try:
        data = request.get_json()
        student_id = data.get('studentId')
        student_name = data.get('studentName')
        images_base64 = data.get('images')

        if not student_id or not student_name or not images_base64:
            logger.error("[ADD_STUDENT]: Missing studentId, studentName, or images")
            return jsonify({"error": "Missing studentId, studentName, or images"}), 400

        if len(images_base64) < 7:
            logger.error(f"[ADD_STUDENT]: Insufficient images for {student_name}: {len(images_base64)} provided")
            return jsonify({"error": "Minimum 7 images required"}), 400

        # Decode base64 images
        images = []
        for idx, img_base64 in enumerate(images_base64):
            try:
                img_data = base64.b64decode(img_base64)
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"[ADD_STUDENT]: Invalid image at index {idx} for {student_name}")
                    continue
                images.append(img)
            except Exception as e:
                logger.error(f"[ADD_STUDENT]: Error decoding image at index {idx} for {student_name}: {str(e)}")
                continue

        if len(images) < 7:
            logger.error(f"[ADD_STUDENT]: Insufficient valid images for {student_name}: {len(images)} decoded")
            return jsonify({"error": f"Only {len(images)} valid images decoded, minimum 7 required"}), 400

        # Add student and embeddings
        success, message = face_service.add_student_with_images(student_id, student_name, images)
        if not success:
            return jsonify({"error": message}), 400

        logger.info(f"[ADD_STUDENT]: Successfully added {student_name} (ID: {student_id})")
        return jsonify({"message": message, "studentId": student_id}), 200

    except Exception as e:
        logger.error(f"[ADD_STUDENT]: Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/add_student', methods=['POST'])
def add_student():
    try:
        data = request.get_json()
        name = data['name']
        image_list = data['images']
        class_id = data.get('classId', '')

        images = []
        for img_data in image_list:
            decoded = base64.b64decode(img_data)
            image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)

        if not images:
            logger.error("[ADD_STUDENT]: No valid images provided")
            return jsonify({"error": "No valid images provided"}), 400

        # Thêm học sinh vào Firestore
        student_data = {
            "studentName": name,
            "classId": class_id,
            "email": data.get('email', ''),
            "gender": data.get('gender', ''),
            "phone": data.get('phone', ''),
            "status": "active",
            "studentCode": data.get('studentCode', ''),
            "userId": data.get('userId', ''),
            "embedding": []  # Sẽ cập nhật sau
        }
        student_id = firestore_service.add_student(student_data)

        # Thêm embedding
        if not face_service.add_face(name, images, class_id, student_id):
            logger.error("[ADD_STUDENT]: Failed to generate face embeddings")
            return jsonify({"error": "Failed to generate face embeddings"}), 400

        # Thêm vào StudentClasses
        firestore_service.db.collection('StudentClasses').add({
            'classId': class_id,
            'studentId': student_id,
            'joinedAt': firestore.SERVER_TIMESTAMP
        })

        logger.info(f"[ADD_STUDENT]: Added student {name} with ID {student_id} in class {class_id}")
        return jsonify({"message": f"Added {name} successfully", "studentId": student_id}), 200
    except Exception as e:
        logger.error(f"[ADD_STUDENT]: Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    logger.info("[HOME]: Accessed Smart Classroom API")
    return jsonify({"message": "Welcome to Smart Classroom API"}), 200

if __name__ == "__main__":
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    scheduler.start()
    schedule_all_classes()
    setup_firestore_listener()
    ngrok.set_auth_token("2u0rqcmZCraUTiXy8NtAyk9wVhT_6YNty8g1YBdcEwDM9FUgV")
    public_url = ngrok.connect(5001, bind_tls=True).public_url
    logger.info(f"[START_SERVER]: Server running at {public_url}")
    app.run(host="0.0.0.0", port=5001)