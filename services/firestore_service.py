import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBARE_CREDENTIALS
from datetime import datetime

class FirestoreService:
    def __init__(self):
        cred = credentials.Certificate(FIREBARE_CREDENTIALS)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def get_student_by_name(self, student_name):
       docs = self.db.collection('Students').where("studentName", "==", student_name).stream()
       for doc in docs:
           return doc.id, doc.to_dict()['classId']
       return None, None

    def save_student_emotion(self, student_id, class_id, emotion_percentages):
       timestamp = datetime.now()
       doc_ref = self.db.collection('StudentEmotionStats').document()
       doc_ref.set({
           'studentEmotionStatsId': doc_ref.id,
           'classId': class_id,
           'studentId': student_id,
           'angry': emotion_percentages.get('Angry', 0),
           'happy': emotion_percentages.get('Happy', 0),
           'neutral': emotion_percentages.get('Neutral', 0),
           'sad': emotion_percentages.get('Sad', 0),
           'surprise': emotion_percentages.get('Surprise', 0),
           'fear': emotion_percentages.get('Fear', 0),
           'createAt': timestamp
       })
    def update_class_emotion_stats(self, class_id, start_time, end_time):
       docs = self.db.collection('StudentEmotionStats').where('classId', '==', class_id)\
           .where('createAt', '>=', start_time).where('createAt', '<=', end_time).stream()
       student_stats = list(docs)
       if not student_stats:
           return
       total_students = len(student_stats)
       avg_emotion = {
           'happy': 0, 'neutral': 0, 'sad': 0, 'surprise': 0, 'angry': 0, 'fear':0
       }
       for doc in student_stats:
           data = doc.to_dict()
           for emotion in avg_emotion.keys():
               avg_emotion[emotion] += data[emotion] / total_students
       timestamp = datetime.now()
       doc_ref =  self.db.collection('ClassEmotionStats').document()
       doc_ref.set({
           'classEmotionStatsId': doc_ref.id,
           'classId': class_id,
           'angry': avg_emotion['angry'],
           'happy': avg_emotion['happy'],
           'neutral': avg_emotion['neutral'],
           'sad': avg_emotion['sad'],
           'surprise': avg_emotion['surprise'],
           'fear': avg_emotion['fear'],
           'createAt': timestamp
       })
    def add_student(self, student_data):
       doc_ref = self.db.collection('Students').document()
       student_data['studentId'] = doc_ref.id
       doc_ref.set(student_data)
       return student_data['studentId']
