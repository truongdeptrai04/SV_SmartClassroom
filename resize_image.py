import os
import cv2
import numpy as np
import logging
import shutil
import time
# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
                    handlers=[logging.FileHandler('face_resize.log'), logging.StreamHandler()])
logger = logging.getLogger('FaceResize')

# Thư mục chứa ảnh khuôn mặt đã trích xuất và thư mục đích
SOURCE_DIR = '/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/extracted_faces'  # Thư mục chứa các ảnh đã cắt
OUTPUT_BASE_DIR = '/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/resized_faces'  # Thư mục gốc để lưu ảnh đã resize

# Kích thước chuẩn của RAF-DB
TARGET_SIZE = (100, 100)  # (width, height)

# Tạo các thư mục đích nếu chưa tồn tại
EMOTIONS = ['Neutral', 'Happy', 'Angry', 'fear', 'Sad', 'Surprise']  # Danh sách cảm xúc, thêm nếu cần
for emotion in EMOTIONS:
    output_dir = os.path.join(OUTPUT_BASE_DIR, emotion)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

def resize_and_save_face(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        start_time = time.time()

        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return

        # Lấy cảm xúc từ tên file (ví dụ: Happy_face_... → Happy)
        filename = os.path.basename(image_path)
        emotion = filename.split('_')[0]  # Lấy phần đầu tiên trước dấu '_'
        if emotion not in EMOTIONS:
            logger.warning(f"Unknown emotion {emotion} in {filename}, skipping")
            return

        # Resize ảnh về kích thước 100x100
        resized_image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        # Tạo đường dẫn đích
        output_filename = f"{os.path.splitext(filename)[0]}_resized.jpg"  # Giữ nguyên tên gốc + "_resized"
        output_path = os.path.join(OUTPUT_BASE_DIR, emotion, output_filename)

        # Lưu ảnh
        cv2.imwrite(output_path, resized_image)
        logger.info(f"Saved resized image to {output_path}")

        total_time = time.time() - start_time
        logger.info(f"Completed processing {filename} (total time: {total_time:.2f}s)")

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")

def main():
    logger.info(f"Starting face resizing from {SOURCE_DIR}")
    start_time = time.time()

    # Duyệt qua tất cả các file ảnh trong thư mục nguồn
    for filename in os.listdir(SOURCE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(SOURCE_DIR, filename)
            resize_and_save_face(image_path)

    total_time = time.time() - start_time
    logger.info(f"Finished face resizing from {SOURCE_DIR} (total time: {total_time:.2f}s)")

if __name__ == "__main__":
    main()