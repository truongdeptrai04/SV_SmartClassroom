import logging
import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_CREDENTIALS
from datetime import datetime
import pytz
from google.cloud.firestore_v1.base_query import FieldFilter

class FirestoreService:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        try:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')
            self.logger.info("Initialized Firestore client")
        except Exception as e:
            self.logger.error(f"Error initializing Firestore: {str(e)}")
            raise

    def get_class_by_time(self, current_time, current_day):
        self.logger.info(f"Checking classes for time: {current_time}, day: {current_day}")
        docs = self.db.collection('Classes').stream()
        current_time_vn = current_time.astimezone(self.VN_TIMEZONE)
        for doc in docs:
            class_data = doc.to_dict()
            start_time = datetime(
                class_data['startTime'].year,
                class_data['startTime'].month,
                class_data['startTime'].day,
                class_data['startTime'].hour,
                class_data['startTime'].minute,
                class_data['startTime'].second,
                class_data['startTime'].microsecond,
                tzinfo=class_data['startTime'].tzinfo
            ).astimezone(self.VN_TIMEZONE)
            end_time = datetime(
                class_data['endTime'].year,
                class_data['endTime'].month,
                class_data['endTime'].day,
                class_data['endTime'].hour,
                class_data['endTime'].minute,
                class_data['endTime'].second,
                class_data['endTime'].microsecond,
                tzinfo=class_data['endTime'].tzinfo
            ).astimezone(self.VN_TIMEZONE)
            day_of_week = class_data['dayOfWeek']
            self.logger.info(
                f"Class {doc.id}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')} (VN), dayOfWeek: {day_of_week}")

            current_hm = current_time_vn.hour * 60 + current_time_vn.minute
            start_hm = start_time.hour * 60 + start_time.minute
            end_hm = end_time.hour * 60 + end_time.minute

            if day_of_week.lower() == current_day.lower():
                if end_hm < start_hm:
                    if current_hm >= start_hm or current_hm <= end_hm:
                        self.logger.info(f"Matched class {doc.id} for time {current_time_vn.strftime('%H:%M')} on {day_of_week}")
                        return doc.id, class_data
                else:
                    if start_hm <= current_hm <= end_hm:
                        self.logger.info(f"Matched class {doc.id} for time {current_time_vn.strftime('%H:%M')} on {day_of_week}")
                        return doc.id, class_data
        self.logger.info("No matching class found")
        return None, None

    def get_student_by_name(self, student_name):
        try:
            docs = self.db.collection('Students').where(filter=FieldFilter("studentName", "==", student_name)).stream()
            for doc in docs:
                self.logger.info(f"Found student {student_name} with ID {doc.id}")
                return doc.id, doc.to_dict().get('classId', '')
            self.logger.warning(f"Student {student_name} not found")
            return None, None
        except Exception as e:
            self.logger.error(f"Error getting student {student_name}: {str(e)}")
            return None, None

    def save_student_emotion(self, student_id, class_id, emotion_percentages):
        try:
            timestamp = datetime.now(self.VN_TIMEZONE)
            class_data = self.db.collection('Classes').document(class_id).get().to_dict()
            start_time = timestamp.replace(
                hour=class_data['startTime'].hour,
                minute=class_data['startTime'].minute,
                second=0, microsecond=0
            )
            end_time = timestamp.replace(
                hour=class_data['endTime'].hour,
                minute=class_data['endTime'].minute,
                second=0, microsecond=0
            )

            doc_ref = self.db.collection('StudentEmotionStats').document()
            doc_ref.set({
                'documentId': doc_ref.id,
                'classId': class_id,
                'studentId': student_id,
                'angry': float(emotion_percentages.get('Angry', 0)),
                'fear': float(emotion_percentages.get('Fear', 0)),
                'happy': float(emotion_percentages.get('Happy', 0)),
                'neutral': float(emotion_percentages.get('Neutral', 0)),
                'sad': float(emotion_percentages.get('Sad', 0)),
                'surprise': float(emotion_percentages.get('Surprise', 0)),
                'totalDetections': 1,
                'createAt': timestamp,
                'startTime': start_time,
                'endTime': end_time
            })
            self.logger.info(
                f"Saved StudentEmotionStats for student {student_id} in class {class_id} at {timestamp}")
        except Exception as e:
            self.logger.error(f"Error saving student emotion for student {student_id} in class {class_id}: {str(e)}")

    def generate_student_session_reports(self, class_id, start_time, end_time):
        try:
            start_time_vn = start_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
            end_time_vn = end_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
            current_time = datetime.now(self.VN_TIMEZONE).replace(microsecond=0)

            class_doc = self.db.collection('Classes').document(class_id).get()
            if not class_doc.exists:
                self.logger.error(f"Class {class_id} not found")
                return []

            class_data = class_doc.to_dict()
            expected_day_of_week = class_data['dayOfWeek'].lower()
            expected_start_hm = class_data['startTime'].hour * 60 + class_data['startTime'].minute
            expected_end_hm = class_data['endTime'].hour * 60 + class_data['endTime'].minute

            self.logger.info(
                f"Generating StudentSessionEmotionReports for class {class_id} from {start_time_vn} to {end_time_vn} (VN)")

            docs = self.db.collection('StudentEmotionStats') \
                .where(filter=FieldFilter('classId', '==', class_id)).stream()

            student_stats = {}
            for doc in docs:
                data = doc.to_dict()
                stat_start_time = datetime(
                    data['startTime'].year,
                    data['startTime'].month,
                    data['startTime'].day,
                    data['startTime'].hour,
                    data['startTime'].minute,
                    data['startTime'].second,
                    data['startTime'].microsecond,
                    tzinfo=data['startTime'].tzinfo
                ).astimezone(self.VN_TIMEZONE)
                stat_end_time = datetime(
                    data['endTime'].year,
                    data['endTime'].month,
                    data['endTime'].day,
                    data['endTime'].hour,
                    data['endTime'].minute,
                    data['endTime'].second,
                    data['endTime'].microsecond,
                    tzinfo=data['endTime'].tzinfo
                ).astimezone(self.VN_TIMEZONE)
                stat_day_of_week = stat_start_time.strftime('%A').lower()

                stat_start_hm = stat_start_time.hour * 60 + stat_start_time.minute
                stat_end_hm = stat_end_time.hour * 60 + stat_end_time.minute

                if (stat_day_of_week == expected_day_of_week and
                        stat_start_hm >= expected_start_hm and
                        stat_end_hm <= expected_end_hm):
                    student_id = data['studentId']
                    if student_id not in student_stats:
                        student_stats[student_id] = {
                            'emotions': {'angry': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0},
                            'total_detections': 0
                        }
                    for emotion in student_stats[student_id]['emotions']:
                        student_stats[student_id]['emotions'][emotion] += data[emotion]
                    student_stats[student_id]['total_detections'] += data['totalDetections']
                    self.logger.info(
                        f"Processed StudentEmotionStats for student {student_id}: createAt={data['createAt']}")

            if not student_stats:
                self.logger.info(
                    f"No StudentEmotionStats found for class {class_id} with expected dayOfWeek {expected_day_of_week}")
                return []

            timestamp = datetime.now(self.VN_TIMEZONE)
            report_ids = []
            for student_id, stats in student_stats.items():
                total_detections = stats['total_detections']
                if total_detections == 0:
                    self.logger.warning(f"No valid detections for student {student_id} in class {class_id}")
                    continue

                avg_emotions = {
                    emotion: (value / total_detections) for emotion, value in stats['emotions'].items()
                }
                total = sum(avg_emotions.values())
                if total > 0:
                    avg_emotions = {k: (v / total) * 100 for k, v in avg_emotions.items()}

                doc_ref = self.db.collection('StudentSessionEmotionReports').document()
                doc_ref.set({
                    'documentId': doc_ref.id,
                    'studentId': student_id,
                    'classId': class_id,
                    'startTime': start_time_vn,
                    'endTime': end_time_vn,
                    'createAt': timestamp,
                    'angry': avg_emotions['angry'],
                    'fear': avg_emotions['fear'],
                    'happy': avg_emotions['happy'],
                    'neutral': avg_emotions['neutral'],
                    'sad': avg_emotions['sad'],
                    'surprise': avg_emotions['surprise'],
                    'totalDetections': total_detections
                })
                report_ids.append(doc_ref.id)
                self.logger.info(
                    f"Saved StudentSessionEmotionReport for student {student_id} in class {class_id} with {total_detections} detections")

            return report_ids
        except Exception as e:
            self.logger.error(f"Error generating student session reports for class {class_id}: {str(e)}")
            return []

    def update_class_emotion_stats(self, class_id, start_time, end_time):
        try:
            start_time_vn = start_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
            end_time_vn = end_time.astimezone(self.VN_TIMEZONE).replace(microsecond=0)
            current_time = datetime.now(self.VN_TIMEZONE).replace(microsecond=0)

            class_doc = self.db.collection('Classes').document(class_id).get()
            if not class_doc.exists:
                self.logger.error(f"Class {class_id} not found")
                return
            class_data = class_doc.to_dict()
            expected_day_of_week = class_data['dayOfWeek'].lower()

            self.logger.info(
                f"Updating ClassEmotionStats for class {class_id} from {start_time_vn} to {end_time_vn} (VN)")

            # Generate student session reports
            report_ids = self.generate_student_session_reports(class_id, start_time, end_time)
            if not report_ids:
                self.logger.info(f"No StudentSessionEmotionReports generated for class {class_id}")
                return

            # Aggregate emotions from StudentSessionEmotionReports
            docs = self.db.collection('StudentSessionEmotionReports') \
                .where(filter=FieldFilter('classId', '==', class_id)) \
                .where(filter=FieldFilter('startTime', '==', start_time_vn)) \
                .where(filter=FieldFilter('endTime', '==', end_time_vn)).stream()

            total_emotions = {
                'angry': 0, 'fear': 0, 'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0
            }
            total_detections = 0
            total_students = 0

            for doc in docs:
                data = doc.to_dict()
                for emotion in total_emotions:
                    total_emotions[emotion] += data[emotion]
                total_detections += data['totalDetections']
                total_students += 1
                self.logger.info(
                    f"Processed StudentSessionEmotionReport for student {data['studentId']}: emotions={data}")

            if total_detections == 0:
                self.logger.error(f"No valid detections for class {class_id}")
                return

            avg_emotions = {
                emotion: (value / total_students) for emotion, value in total_emotions.items()
            }
            total = sum(avg_emotions.values())
            if total > 0:
                avg_emotions = {k: (v / total) * 100 for k, v in avg_emotions.items()}

            timestamp = datetime.now(self.VN_TIMEZONE)
            doc_ref = self.db.collection('ClassEmotionStats').document()
            doc_ref.set({
                'classId': class_id,
                'createAt': timestamp,
                'startTime': start_time_vn,
                'endTime': end_time_vn,
                'angry': avg_emotions['angry'],
                'fear': avg_emotions['fear'],
                'happy': avg_emotions['happy'],
                'neutral': avg_emotions['neutral'],
                'sad': avg_emotions['sad'],
                'surprise': avg_emotions['surprise'],
                'totalDetectedStudents': total_students,
                'totalDetections': total_detections
            })
            self.logger.info(
                f"Updated ClassEmotionStats for class {class_id} with {total_students} students, {total_detections} detections")

            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            dominant_percentage = avg_emotions[dominant_emotion]
            content = f"Cảm xúc chính của lớp {class_id} là {dominant_emotion} ({dominant_percentage:.2f}%) vào lúc {timestamp.strftime('%Y-%m-%d %H:%M:%S%z')}"

            alert_ref = self.db.collection('Alerts').document()
            alert_ref.set({
                'classId': class_id,
                'content': content,
                'timestamp': timestamp,
                'title': 'Report'
            })
            self.logger.info(f"Added Alert for class {class_id}: {content}")

        except Exception as e:
            self.logger.error(f"Error updating class emotion stats for class {class_id}: {str(e)}")

    def add_student(self, student_data):
        try:
            doc_ref = self.db.collection('Students').document()
            student_data['studentId'] = doc_ref.id
            doc_ref.set(student_data)
            self.logger.info(f"Added student {student_data['studentId']}")
            return student_data['studentId']
        except Exception as e:
            self.logger.error(f"Error adding student: {str(e)}")
            return None