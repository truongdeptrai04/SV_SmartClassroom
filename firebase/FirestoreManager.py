import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import uuid

class FirestoreManager:
    def __init__(self, credentials_path):
        # Khởi tạo Firebase
        cred = credentials.Certificate(credentials_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def generate_document_id(self):
        return str(uuid.uuid4())

    def parse_timestamp(self, timestamp_str):
        """Chuyển chuỗi ngày giờ thành Firestore Timestamp"""
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            return firestore.SERVER_TIMESTAMP if timestamp_str.lower() == "now" else dt
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS' or 'now': {e}")

    # Alerts
    def add_alert(self, class_id, content, timestamp_str, title):
        doc_id = self.generate_document_id()
        data = {
            "classId": class_id,
            "content": content,
            "timestamp": self.parse_timestamp(timestamp_str),
            "title": title
        }
        self.db.collection("Alerts").document(doc_id).set(data)
        print(f"Added Alert: {doc_id}")
        return doc_id

    def delete_alert(self, doc_id):
        self.db.collection("Alerts").document(doc_id).delete()
        print(f"Deleted Alert: {doc_id}")

    # ClassEmotionStats
    def add_class_emotion_stats(self, class_id, angry, fear, happy, neutral, sad, surprise, total_detected, create_at, end_time, start_time):
        doc_id = self.generate_document_id()
        data = {
            "classId": class_id,
            "angry": float(angry),
            "fear": float(fear),
            "happy": float(happy),
            "neutral": float(neutral),
            "sad": float(sad),
            "surprise": float(surprise),
            "totalDetectedStudents": int(total_detected),
            "createAt": self.parse_timestamp(create_at),
            "endTime": self.parse_timestamp(end_time),
            "startTime": self.parse_timestamp(start_time)
        }
        self.db.collection("ClassEmotionStats").document(doc_id).set(data)
        print(f"Added ClassEmotionStats: {doc_id}")
        return doc_id

    def delete_class_emotion_stats(self, doc_id):
        self.db.collection("ClassEmotionStats").document(doc_id).delete()
        print(f"Deleted ClassEmotionStats: {doc_id}")

    # Classes
    def add_class(self, class_id, class_name, day_of_week, description, end_time, start_time, user_id):
        doc_id = self.generate_document_id()
        data = {
            "classId": class_id,
            "className": class_name,
            "dayOfWeek": day_of_week,
            "description": description,
            "endTime": self.parse_timestamp(end_time),
            "startTime": self.parse_timestamp(start_time),
            "userId": user_id
        }
        self.db.collection("Classes").document(doc_id).set(data)
        print(f"Added Class: {doc_id}")
        return doc_id

    def delete_class(self, doc_id):
        self.db.collection("Classes").document(doc_id).delete()
        print(f"Deleted Class: {doc_id}")

    # StudentClasses
    def add_student_class(self, class_id, student_id, joined_at):
        doc_id = self.generate_document_id()
        data = {
            "classId": class_id,
            "studentId": student_id,
            "joinedAt": joined_at
        }
        self.db.collection("StudentClasses").document(doc_id).set(data)
        print(f"Added StudentClass: {doc_id}")
        return doc_id

    def delete_student_class(self, doc_id):
        self.db.collection("StudentClasses").document(doc_id).delete()
        print(f"Deleted StudentClass: {doc_id}")

    # StudentEmotionStats
    def add_student_emotion_stats(self, class_id, angry, fear, happy, neutral, sad, surprise, total_detection, create_at, end_time, start_time, student_id):
        doc_id = self.generate_document_id()
        data = {
            "classId": class_id,
            "angry": float(angry),
            "fear": float(fear),
            "happy": float(happy),
            "neutral": float(neutral),
            "sad": float(sad),
            "surprise": float(surprise),
            "totalDetection": int(total_detection),
            "createAt": self.parse_timestamp(create_at),
            "endTime": self.parse_timestamp(end_time),
            "startTime": self.parse_timestamp(start_time),
            "studentId": student_id
        }
        self.db.collection("StudentEmotionStats").document(doc_id).set(data)
        print(f"Added StudentEmotionStats: {doc_id}")
        return doc_id

    def delete_student_emotion_stats(self, doc_id):
        self.db.collection("StudentEmotionStats").document(doc_id).delete()
        print(f"Deleted StudentEmotionStats: {doc_id}")

    # Students
    def add_student(self, avatar_url, date_of_birth, email, gender, notes, phone, status, student_code, student_name, user_id):
        doc_id = self.generate_document_id()
        data = {
            "avatarUrl": avatar_url,
            "dateOfBirth": date_of_birth,
            "email": email,
            "gender": gender,
            "notes": notes,
            "phone": phone,
            "status": status,
            "studentCode": student_code,
            "studentName": student_name,
            "userId": user_id
        }
        self.db.collection("Students").document(doc_id).set(data)
        print(f"Added Student: {doc_id}")
        return doc_id

    def delete_student(self, doc_id):
        self.db.collection("Students").document(doc_id).delete()
        print(f"Deleted Student: {doc_id}")

    # User
    def add_user(self, email, role, username):
        doc_id = self.generate_document_id()
        data = {
            "email": email,
            "role": role,
            "username": username
        }
        self.db.collection("User").document(doc_id).set(data)
        print(f"Added User: {doc_id}")
        return doc_id

    def delete_user(self, doc_id):
        self.db.collection("User").document(doc_id).delete()
        print(f"Deleted User: {doc_id}")