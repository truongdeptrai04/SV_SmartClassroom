import os
import cv2
import numpy as np
from deepface import DeepFace
from firebase_admin import firestore
import firebase_admin
from config import FIREBASE_CREDENTIALS
from google.cloud.firestore_v1.base_query import FieldFilter
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Firestore
if not firebase_admin._apps:
    firebase_admin.initialize_app(firebase_admin.credentials.Certificate(FIREBASE_CREDENTIALS))
db = firestore.client()

# Hàm resize ảnh
def resize_image(img, max_size=640):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img

# Hàm chuẩn hóa embedding
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def update_student_embedding_manual(students_folders, class_id='cl_012'):
    """Cập nhật thủ công embedding cho các học sinh cụ thể trong Firestore."""
    try:
        logger.info(f"Initialized Firestore client, project: {firebase_admin.get_app().project_id}")

        for student_id, image_folder_path in students_folders:
            try:
                # Kiểm tra thư mục tồn tại
                if not os.path.exists(image_folder_path):
                    logger.error(f"Thư mục {image_folder_path} không tồn tại")
                    continue

                embeddings = []
                # Duyệt qua các ảnh trong thư mục
                for img_name in os.listdir(image_folder_path):
                    img_path = os.path.join(image_folder_path, img_name)
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        logger.debug(f"Bỏ qua file không phải ảnh: {img_path}")
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        logger.error(f"Không đọc được ảnh {img_path}")
                        continue

                    img = resize_image(img)

                    try:
                        emb = DeepFace.represent(
                            img,
                            model_name='Facenet',
                            detector_backend='retinaface',
                            enforce_detection=False
                        )
                        if emb and emb[0]['embedding']:
                            embedding = normalize_embedding(np.array(emb[0]['embedding']))
                            embeddings.append(embedding)
                            logger.info(f"Tạo embedding thành công cho {img_path}")
                        else:
                            logger.warning(f"Không phát hiện khuôn mặt trong {img_path}")
                    except Exception as e:
                        logger.error(f"Lỗi tạo embedding cho {img_path}: {str(e)}")

                if embeddings:
                    # Tính embedding trung bình
                    avg_embedding = np.mean(embeddings, axis=0)
                    avg_embedding = normalize_embedding(avg_embedding)
                    logger.info(f"Tạo embedding trung bình cho student ID: {student_id}, shape: {avg_embedding.shape}")

                    # Cập nhật Firestore
                    student_ref = db.collection('Students').document(student_id)
                    student_doc = student_ref.get()

                    if student_doc.exists:
                        # Cập nhật embedding cho học sinh hiện có
                        student_data = student_doc.to_dict()
                        student_name = student_data.get('studentName', student_id)
                        student_ref.update({
                            'embedding': avg_embedding.tolist(),
                            'studentName': student_name
                        })
                        logger.info(f"Đã cập nhật embedding cho {student_name} (ID: {student_id})")
                    else:
                        # Tạo học sinh mới
                        student_ref.set({
                            'studentId': student_id,
                            'studentName': student_id,  # Có thể thay bằng tên thực nếu có
                            'embedding': avg_embedding.tolist(),
                            'email': f"{student_id.lower()}@example.com",
                            'gender': 'Unknown',
                            'phone': '',
                            'studentCode': f"STU_{student_id}",
                            'userId': f"user_{student_id}",
                            'status': 'active'
                        })
                        logger.info(f"Tạo student mới và cập nhật embedding cho ID: {student_id}")

                    # Cập nhật hoặc tạo StudentClasses
                    student_classes = db.collection('StudentClasses') \
                        .where(filter=FieldFilter('studentId', '==', student_id)) \
                        .where(filter=FieldFilter('classId', '==', class_id)) \
                        .stream()
                    if not any(student_classes):
                        db.collection('StudentClasses').add({
                            'studentId': student_id,
                            'classId': class_id,
                            'joinedAt': firestore.SERVER_TIMESTAMP
                        })
                        logger.info(f"Thêm liên kết StudentClasses cho {student_id} (class: {class_id})")
                    else:
                        logger.debug(f"Liên kết StudentClasses đã tồn tại cho {student_id} (class: {class_id})")

                else:
                    logger.error(f"Không tạo được embedding cho student ID: {student_id}")

            except Exception as e:
                logger.error(f"Lỗi xử lý student ID: {student_id}, folder: {image_folder_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Lỗi khởi tạo Firestore: {str(e)}")
        raise

if __name__ == "__main__":
    # Danh sách các cặp {studentId, image_folder_path}
    students_folders = [
        ("std_004", "/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/team_data/Dan"),
        ("std_003", "/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/team_data/Dung"),
        ("std_005", "/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/team_data/Trong"),
    ]

    update_student_embedding_manual(students_folders, class_id='cl_012')