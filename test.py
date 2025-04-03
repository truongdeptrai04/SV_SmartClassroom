import cv2
import numpy as np
import face_recognition
import pickle
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

FACE_ENCODINGS_PATH = "face_encodings.pkl"
MODEL_PATH = "models/fer2013_model.h5"
HAAR_CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

try:
    with open(FACE_ENCODINGS_PATH, 'rb') as f:
        known_faces = pickle.load(f)
    known_encodings = list(known_faces.values())
    known_names = list(known_faces.keys())
except FileNotFoundError:
    print("Không tìm thấy face_encodings.pkl. Vui lòng tạo file này trước!")
    exit()

try:
    model = load_model(MODEL_PATH)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print("Model input shape:", model.input_shape)
except Exception as e:
    print(f"Không thể tải mô hình: {e}")
    exit()

face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    print("Không thể tải Haar Cascade. Kiểm tra đường dẫn!")
    exit()


def find_working_camera():
    """Tìm camera hoạt động"""
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            print(f"Đã tìm thấy camera với ID {i}")
            return cap
        cap.release()
    return None


def recognize_face(image, tolerance=0.5):
    """Nhận diện khuôn mặt với ngưỡng tolerance"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image, model="hog")  # Dùng HOG để nhanh hơn
    if not encodings:
        return "Unknown", 0.0

    distances = face_recognition.face_distance(known_encodings, encodings[0])
    min_distance = np.min(distances)

    if min_distance < tolerance:
        best_match_index = np.argmin(distances)
        name = known_names[best_match_index]
        confidence = (1 - min_distance) * 100
        return name, confidence
    return "Unknown", 0.0


def preprocess_face(image):
    """Chuẩn hóa và căn chỉnh khuôn mặt cho FER2013"""
    gray_face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(gray_face)
    face = cv2.resize(gray_face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=(-1, 0))
    return face


def detect_emotion(image):
    """Dự đoán cảm xúc với phần trăm độ tin cậy"""
    try:
        face = preprocess_face(image)
        prediction = model.predict(face, verbose=0)[0]
        total = sum(prediction)
        if total == 0:
            return "Error", 0.0

        emotion_percentages = {emotion: (prob / total) * 100 for emotion, prob in zip(emotions, prediction)}

        dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
        confidence = emotion_percentages[dominant_emotion]

        return dominant_emotion, confidence
    except Exception as e:
        print(f"Lỗi dự đoán cảm xúc: {e}")
        return "Error", 0.0


def main():
    cap = cv2.VideoCapture(0)
    if not cap:
        print("Không tìm thấy camera nào hoạt động! Kiểm tra quyền hoặc kết nối.")
        return

    frame_skip = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình từ camera!")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            cv2.imshow("Face Recognition and Emotion Detection", frame)
            continue

        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            student_name, face_confidence = recognize_face(face_roi, tolerance=0.5)
            emotion, emotion_confidence = detect_emotion(face_roi)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{student_name} ({face_confidence:.1f}%) - {emotion} ({emotion_confidence:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition and Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()