import cv2
import base64
import requests
import time
import os
import logging
from datetime import datetime
import pytz

# Cấu hình
SERVER_URL = "https://5af7-58-187-196-90.ngrok-free.app/process_image"  # Thay bằng URL ngrok thực tế
IMAGE_DIR = "camera_images"
VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')
INTERVAL_SECONDS = 15

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('test_camera.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_camera():
    """Khởi tạo camera"""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logger.error("Không thể mở camera")
        raise ValueError("Không thể mở camera")
    logger.info("Camera đã được khởi tạo")
    return cap

def capture_image(cap):
    """Chụp ảnh từ camera"""
    ret, frame = cap.read()
    if not ret:
        logger.error("Không thể chụp ảnh từ camera")
        return None
    return frame

def save_image(image, timestamp):
    """Lưu ảnh để debug"""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    filename = os.path.join(IMAGE_DIR, f"camera_{timestamp}.jpg")
    cv2.imwrite(filename, image)
    logger.info(f"Đã lưu ảnh tại {filename}")
    return filename

def encode_image(image):
    """Mã hóa ảnh thành base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def send_image_to_server(img_base64):
    """Gửi ảnh lên server"""
    payload = {"image": img_base64}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        logger.info(f"Phản hồi từ server: {response.status_code} - {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Lỗi khi gửi request: {str(e)}")
        return None

def main():
    """Chương trình chính"""
    try:
        # Khởi tạo camera
        cap = setup_camera()

        while True:
            # Lấy thời gian hiện tại
            timestamp = datetime.now(VN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
            logger.info(f"Bắt đầu chu kỳ chụp ảnh tại {timestamp}")

            # Chụp ảnh
            image = capture_image(cap)
            if image is None:
                time.sleep(INTERVAL_SECONDS)
                continue

            # Lưu ảnh để debug
            save_image(image, timestamp)

            # Mã hóa ảnh
            img_base64 = encode_image(image)
            logger.info("Đã mã hóa ảnh thành base64")

            # Gửi lên server
            response = send_image_to_server(img_base64)
            if response:
                logger.info(f"Kết quả nhận diện: {response.get('results', [])}")
            else:
                logger.warning("Không nhận được phản hồi từ server")

            # Đợi 15 giây
            logger.info(f"Đợi {INTERVAL_SECONDS} giây trước khi chụp tiếp...")
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Dừng chương trình bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi chương trình: {str(e)}")
    finally:
        # Giải phóng camera
        if 'cap' in locals():
            cap.release()
            logger.info("Đã giải phóng camera")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()