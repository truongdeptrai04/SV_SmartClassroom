import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_CREDENTIALS
import logging
import base64
import os
import cv2
import numpy as np

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_and_compress_image(image_path, max_size=256, quality=85):
    """Resize và nén ảnh để đảm bảo base64 nhỏ hơn 1MB."""
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to read image {image_path}")
        raise ValueError(f"Invalid image {image_path}")

    # Resize ảnh
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Nén ảnh
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode('.jpg', img, encode_param)
    if not success:
        logger.error(f"Failed to compress image {image_path}")
        raise ValueError(f"Failed to compress image {image_path}")

    base64_string = base64.b64encode(buffer).decode('utf-8')
    base64_size = len(base64_string)

    # Kiểm tra kích thước
    MAX_FIRESTORE_SIZE = 1048487  # bytes
    if base64_size > MAX_FIRESTORE_SIZE:
        logger.warning(
            f"Base64 size {base64_size} bytes for {image_path} exceeds Firestore limit {MAX_FIRESTORE_SIZE} bytes")
        # Thử giảm chất lượng hơn
        for q in range(quality - 10, 50, -10):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            success, buffer = cv2.imencode('.jpg', img, encode_param)
            if not success:
                continue
            base64_string = base64.b64encode(buffer).decode('utf-8')
            base64_size = len(base64_string)
            if base64_size <= MAX_FIRESTORE_SIZE:
                logger.info(f"Compressed image {image_path} with quality {q}, base64 length: {base64_size}")
                return base64_string
        logger.error(f"Cannot compress {image_path} under {MAX_FIRESTORE_SIZE} bytes")
        raise ValueError(f"Base64 size {base64_size} exceeds Firestore limit")

    logger.info(f"Processed image {image_path}, base64 length: {base64_size}")
    return base64_string


def add_student_avatar_manual(students_images):
    """Thêm thủ công avatarUrl (base64) cho các học sinh cụ thể trong Firestore."""
    try:
        # Khởi tạo Firebase app
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(FIREBASE_CREDENTIALS))
        db = firestore.client()
        logger.info(f"Initialized Firestore client, project: {firebase_admin.get_app().project_id}")

        updated_count = 0
        failed_count = 0

        # Duyệt qua danh sách studentId và image_path
        for student_id, image_path in students_images:
            try:
                # Resize và nén ảnh, chuyển thành base64
                avatar_base64 = resize_and_compress_image(image_path)

                # Cập nhật Firestore
                student_ref = db.collection('Students').document(student_id)
                student_doc = student_ref.get()

                if student_doc.exists:
                    student_ref.update({
                        'avatarUrl': avatar_base64
                    })
                    logger.info(f"Updated avatarUrl for student ID: {student_id}, base64 length: {len(avatar_base64)}")
                    updated_count += 1
                else:
                    logger.error(f"Student ID: {student_id} not found in Firestore")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error processing student ID: {student_id}, image: {image_path}: {str(e)}")
                failed_count += 1

        logger.info(f"Completed: Updated {updated_count} students, failed {failed_count} students")

    except Exception as e:
        logger.error(f"Error initializing Firestore or processing: {str(e)}")
        raise


if __name__ == "__main__":
    # Danh sách các cặp {studentId, image_path}
    students_images = [
        ("std_005", "/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/team_data/Trong/Trong_13.jpg"),
    ]

    add_student_avatar_manual(students_images)