import cv2
import base64
import requests
import time
from datetime import datetime
import pytz
import json

# Cấu hình
SERVER_URL = "https://7745-58-187-196-90.ngrok-free.app/process_image"  # Thay bằng IP của Raspberry Pi
SEND_INTERVAL = 15  # Gửi mỗi 15 giây
VN_TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')  # UTC+7

def capture_and_send():
    # Mở camera (index 0 là camera mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    # Giảm độ phân giải để tăng hiệu suất
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_sent = 0
    try:
        while True:
            # Đọc khung hình từ camera
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc khung hình từ camera")
                break

            # Hiển thị video trực tiếp
            cv2.imshow("Camera", frame)

            # Kiểm tra thời gian để gửi ảnh
            current_time = time.time()
            if current_time - last_sent >= SEND_INTERVAL:
                # Mã hóa ảnh thành base64
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Tạo payload
                payload = {
                    "image": image_base64
                }

                # Gửi yêu cầu POST đến server
                try:
                    vn_time = datetime.now(VN_TIMEZONE)
                    print(f"Gửi ảnh lúc {vn_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    response = requests.post(SERVER_URL, json=payload, timeout=5)
                    print(f"Phản hồi từ server: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"Lỗi khi gửi yêu cầu: {e}")

                last_sent = current_time

            # Thoát nếu nhấn phím 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Giải phóng camera và đóng cửa sổ
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Bắt đầu test camera. Nhấn 'q' để thoát.")
    capture_and_send()