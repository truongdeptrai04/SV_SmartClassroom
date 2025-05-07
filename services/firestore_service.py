import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBARE_CREDENTIALS
from datetime import datetime
import pytz
from google.cloud.firestore_v1.base_query import FieldFilter


class FirestoreService:
    def __init__(self):
        cred = credentials.Certificate(FIREBARE_CREDENTIALS)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')

    def get_class_by_time(self, current_time, current_day):
        print(f"Checking classes for time: {current_time}, day: {current_day}")
        docs = self.db.collection('Classes').stream()
        current_time_vn = current_time.astimezone(self.VN_TIMEZONE)
        for doc in docs:
            class_data = doc.to_dict()
            start_time = class_data['startTime'].astimezone(self.VN_TIMEZONE)
            end_time = class_data['endTime'].astimezone(self.VN_TIMEZONE)
            day_of_week = class_data['dayOfWeek']
            print(
                f"Class {doc.id}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} (VN), dayOfWeek: {day_of_week}")

            # Trích xuất giờ và phút
            current_hm = current_time_vn.hour * 60 + current_time_vn.minute
            start_hm = start_time.hour * 60 + start_time.minute
            end_hm = end_time.hour * 60 + end_time.minute

            # So sánh ngày trong tuần và thời gian
            if day_of_week.lower() == current_day.lower():
                if end_hm < start_hm:  # Trường hợp qua nửa đêm
                    if current_hm >= start_hm or current_hm <= end_hm:
                        print(f"Matched class {doc.id} for time {current_time_vn.strftime('%H:%M')} on {day_of_week}")
                        return doc.id, class_data
                else:  # Trường hợp trong cùng ngày
                    if start_hm <= current_hm <= end_hm:
                        print(f"Matched class {doc.id} for time {current_time_vn.strftime('%H:%M')} on {day_of_week}")
                        return doc.id, class_data
        print("No matching class found")
        return None, None

    def get_student_by_name(self, student_name):
        docs = self.db.collection('Students').where(filter=FieldFilter("studentName", "==", student_name)).stream()
        for doc in docs:
            return doc.id, doc.to_dict().get('classId', '')
        return None, None

    def save_student_emotion(self, student_id, class_id, emotion_percentages):
        timestamp = datetime.now(self.VN_TIMEZONE)
        class_data = self.db.collection('Classes').document(class_id).get().to_dict()
        start_time = timestamp.replace(hour=class_data['startTime'].astimezone(self.VN_TIMEZONE).hour,
                                       minute=class_data['startTime'].astimezone(self.VN_TIMEZONE).minute,
                                       second=0, microsecond=0)
        end_time = timestamp.replace(hour=class_data['endTime'].astimezone(self.VN_TIMEZONE).hour,
                                     minute=class_data['endTime'].astimezone(self.VN_TIMEZONE).minute,
                                     second=0, microsecond=0)

        doc_ref = self.db.collection('StudentEmotionStats').document()
        doc_ref.set({
            'documentId': doc_ref.id,
            'classId': class_id,
            'studentId': student_id,
            'angry': emotion_percentages.get('Angry', 0),
            'happy': emotion_percentages.get('Happy', 0),
            'neutral': emotion_percentages.get('Neutral', 0),
            'sad': emotion_percentages.get('Sad', 0),
            'surprise': emotion_percentages.get('Surprise', 0),
            'fear': emotion_percentages.get('Fear', 0),
            'totalDetections': 1,
            'createAt': timestamp,
            'startTime': start_time,
            'endTime': end_time
        })
        print(
            f"Saved StudentEmotionStats for class {class_id} at {timestamp}, startTime={start_time}, endTime={end_time}")

    def update_class_emotion_stats(self, class_id, start_time, end_time):
        # Chuẩn hóa timestamp
        start_time_vn = start_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
        end_time_vn = end_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
        current_time = datetime.now(self.VN_TIMEZONE).replace(microsecond=0)

        # Lấy thông tin lớp để lấy dayOfWeek
        class_doc = self.db.collection('Classes').document(class_id).get()
        if not class_doc.exists:
            print(f"Class {class_id} not found")
            return
        class_data = class_doc.to_dict()
        expected_day_of_week = class_data['dayOfWeek'].lower()

        # Trích xuất giờ và phút để so sánh
        start_hm = start_time_vn.hour * 60 + start_time_vn.minute
        end_hm = end_time_vn.hour * 60 + end_time_vn.minute
        expected_start_hm = class_data['startTime'].astimezone(self.VN_TIMEZONE).hour * 60 + class_data[
            'startTime'].astimezone(self.VN_TIMEZONE).minute
        expected_end_hm = class_data['endTime'].astimezone(self.VN_TIMEZONE).hour * 60 + class_data[
            'endTime'].astimezone(self.VN_TIMEZONE).minute

        print(
            f"Updating ClassEmotionStats for class {class_id} from {start_time_vn} to {end_time_vn} (VN), current time: {current_time}, expected dayOfWeek: {expected_day_of_week}")

        # Truy vấn StudentEmotionStats chỉ với classId
        docs = self.db.collection('StudentEmotionStats') \
            .where(filter=FieldFilter('classId', '==', class_id)).stream()

        # Lọc thủ công theo giờ, phút và dayOfWeek
        student_stats = []
        for doc in docs:
            data = doc.to_dict()
            stat_start_time = data['startTime'].astimezone(self.VN_TIMEZONE)
            stat_end_time = data['endTime'].astimezone(self.VN_TIMEZONE)
            stat_day_of_week = stat_start_time.strftime('%A').lower()

            # Trích xuất giờ và phút
            stat_start_hm = stat_start_time.hour * 60 + stat_start_time.minute
            stat_end_hm = stat_end_time.hour * 60 + stat_end_time.minute

            # Kiểm tra dayOfWeek và giờ/phút
            if (stat_day_of_week == expected_day_of_week and
                    stat_start_hm >= expected_start_hm and
                    stat_end_hm <= expected_end_hm):
                student_stats.append(doc)
            print(
                f"Found StudentEmotionStats: createAt={data['createAt']}, startTime={data['startTime']}, endTime={data['endTime']}")

        if not student_stats:
            print(
                f"No StudentEmotionStats found for class {class_id} with expected dayOfWeek {expected_day_of_week} and time {start_time_vn.strftime('%H:%M')} to {end_time_vn.strftime('%H:%M')}")
            return

        student_ids = set()
        total_emotions = {
            'angry': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0
        }
        total_detections = 0

        for doc in student_stats:
            data = doc.to_dict()
            student_ids.add(data['studentId'])
            for emotion in total_emotions:
                total_emotions[emotion] += data[emotion]
            total_detections += data['totalDetections']
            print(f"Processed StudentEmotionStats: createAt={data['createAt']}, emotions={data}")

        total_students = len(student_ids)
        if total_detections == 0:
            print(f"No valid detections for class {class_id}")
            return

        # Tính phần trăm trung bình
        avg_emotions = {
            emotion: (value / total_detections) for emotion, value in total_emotions.items()
        }
        # Chuẩn hóa để tổng = 100%
        total = sum(avg_emotions.values())
        if total > 0:
            avg_emotions = {k: (v / total) * 100 for k, v in avg_emotions.items()}

        # Ghi ClassEmotionStats
        timestamp = (datetime.now(self.VN_TIMEZONE))
        doc_ref = self.db.collection('ClassEmotionStats').document()
        doc_ref.set({
            'classId': class_id,
            'createAt': timestamp,
            'startTime': timestamp.replace(hour=start_time_vn.hour, minute=start_time_vn.minute, second=0,
                                           microsecond=0),
            'endTime': timestamp.replace(hour=end_time_vn.hour, minute=end_time_vn.minute, second=0, microsecond=0),
            'angry': avg_emotions['angry'],
            'fear': avg_emotions['fear'],
            'happy': avg_emotions['happy'],
            'neutral': avg_emotions['neutral'],
            'sad': avg_emotions['sad'],
            'surprise': avg_emotions['surprise'],
            'totalDetectedStudents': total_students,
            'totalDetections': total_detections
        })
        print(
            f"Updated ClassEmotionStats for class {class_id} with {total_students} students, {total_detections} detections")

        # Xác định cảm xúc chính
        dominant_emotion = max(avg_emotions, key=avg_emotions.get)
        dominant_percentage = avg_emotions[dominant_emotion]
        content = f"Cảm xúc chính của lớp {class_id} là {dominant_emotion} ({dominant_percentage:.2f}%) vào lúc {timestamp.strftime('%Y-%m-%d %H:%M:%S%z')}"

        # Ghi vào collection Alert
        alert_ref = self.db.collection('Alerts').document()
        alert_ref.set({
            'classId': class_id,
            'content': content,
            'timestamp': timestamp,
            'title': 'Report'
        })
        print(f"Added Alert for class {class_id}: {content}")

    def add_student(self, student_data):
        doc_ref = self.db.collection('Students').document()
        student_data['studentId'] = doc_ref.id
        doc_ref.set(student_data)
        return student_data['studentId']