import os
import cv2
import numpy as np
from deepface import DeepFace
import logging
import time

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
                    handlers=[logging.FileHandler('face_extraction.log'), logging.StreamHandler()])
logger = logging.getLogger('FaceExtraction')

# Thư mục chứa ảnh gốc và thư mục đích để lưu khuôn mặt
SOURCE_DIR = '/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/new'  # Thư mục chứa các thư mục con (Neutral, Happy,...)
OUTPUT_DIR = '/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/extracted_faces'  # Thư mục để lưu khuôn mặt

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

def extract_faces(image_path, emotion_folder):
    try:
        logger.info(f"Processing image: {image_path} (emotion: {emotion_folder})")
        start_time = time.time()

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Phát hiện khuôn mặt sử dụng RetinaFace
        start_detect = time.time()
        face_results = DeepFace.analyze(
            image,
            actions=['emotion'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        detect_time = time.time() - start_detect
        logger.info(f"Detected {len(face_results)} faces in {os.path.basename(image_path)} (took {detect_time:.2f}s)")

        # Cắt và lưu từng khuôn mặt
        for idx, result in enumerate(face_results):
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid face dimensions at ({x}, {y}, {w}, {h}) in {image_path}")
                continue

            face_roi = image[y:y+h, x:x+w]
            output_filename = f"{emotion_folder}_face_{os.path.basename(image_path).split('.')[0]}_{idx}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, face_roi)
            logger.info(f"Saved face {idx} to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"Completed processing {os.path.basename(image_path)} (total time: {total_time:.2f}s)")

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    logger.info(f"Starting face extraction from {SOURCE_DIR}")
    start_time = time.time()

    # Duyệt qua các thư mục con trong SOURCE_DIR
    for emotion_folder in os.listdir(SOURCE_DIR):
        emotion_path = os.path.join(SOURCE_DIR, emotion_folder)
        if os.path.isdir(emotion_path):
            logger.info(f"Processing emotion folder: {emotion_folder}")
            for file in os.listdir(emotion_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(emotion_path, file)
                    extract_faces(image_path, emotion_folder)

    total_time = time.time() - start_time
    logger.info(f"Finished face extraction from {SOURCE_DIR} (total time: {total_time:.2f}s)")

if __name__ == "__main__":
    main()