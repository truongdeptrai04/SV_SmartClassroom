import cv2
import numpy as np
from deepface import DeepFace
from services.firestore_service import FirestoreService
from services.face_recognition_service import FaceRecognitionService
from google.cloud.firestore_v1.base_query import FieldFilter

# Khởi tạo services
fs = FirestoreService()
frs = FaceRecognitionService()

# Đọc ảnh kiểm tra
img = cv2.imread('/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/received_images/received_20250606_205013.jpg')
embedding = DeepFace.represent(img, model_name='Facenet', detector_backend='retinaface', enforce_detection=False)

if embedding:
    embedding = frs.normalize_embedding(np.array(embedding[0]['embedding']))
    student_ids = [sc.to_dict()['studentId'] for sc in fs.db.collection('StudentClasses').where(filter=FieldFilter('classId', '==', 'cl_012')).stream()]
    for student_id in student_ids:
        student = fs.db.collection('Students').document(student_id).get()
        if student.exists:
            stored_embedding = np.array(student.to_dict()['embedding'])
            distance = np.linalg.norm(embedding - stored_embedding)
            print(f"Distance to {student.to_dict()['studentName']} (ID: {student_id}): {distance:.2f}")
else:
    print("Failed to generate embedding")