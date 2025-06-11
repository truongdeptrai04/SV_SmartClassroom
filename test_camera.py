import cv2
import base64
import requests
import time
import os
import logging
from datetime import datetime
import pytz

# Cấu hình
SERVER_URL = "https://bb0c-2402-800-777c-8527-54f-7632-aab3-fc9a.ngrok-free.app/process_image"  # Thay bằng URL ngrok thực tế
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

        # Tạo cửa sổ hiển thị
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Captured Frame", cv2.WINDOW_NORMAL)

        # Thời gian chụp ảnh lần cuối
        last_capture_time = time.time()
        # Biến lưu khung hình đã chụp
        captured_image = None

        while True:
            # Đọc khung hình từ camera
            image = capture_image(cap)
            if image is None:
                logger.warning("Không thể đọc khung hình, thử lại...")
                time.sleep(0.1)
                continue

            # Hiển thị video trực tiếp
            cv2.imshow("Camera Feed", image)

            # Hiển thị khung hình đã chụp (nếu có)
            if captured_image is not None:
                cv2.imshow("Captured Frame", captured_image)

            # Kiểm tra phím nhấn để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Người dùng nhấn 'q' để thoát")
                break

            # Kiểm tra thời gian để chụp và gửi ảnh (mỗi 15 giây)
            current_time = time.time()
            if current_time - last_capture_time >= INTERVAL_SECONDS:
                # Lấy thời gian hiện tại
                timestamp = datetime.now(VN_TIMEZONE).strftime("%Y%m%d_%H%M%S")
                logger.info(f"Bắt đầu chu kỳ chụp ảnh tại {timestamp}")

                # Lưu khung hình vừa chụp
                captured_image = image.copy()  # Sao chép để hiển thị trong cửa sổ riêng

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

                # Cập nhật thời gian chụp ảnh
                last_capture_time = current_time
                logger.info(f"Đợi {INTERVAL_SECONDS} giây trước khi chụp tiếp...")

    except KeyboardInterrupt:
        logger.info("Dừng chương trình bởi người dùng")
    except Exception as e:
        logger.error(f"Lỗi chương trình: {str(e)}")
    finally:
        # Giải phóng camera và đóng cửa sổ
        if 'cap' in locals():
            cap.release()
            logger.info("Đã giải phóng camera")
        cv2.destroyAllWindows()
        logger.info("Đã đóng tất cả cửa sổ OpenCV")

if __name__ == "__main__":
    main()