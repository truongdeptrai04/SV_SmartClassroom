import face_recognition
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config import FACE_ENCODINGS_PATH
from config import VGG_SVM_MODEL_PATH, VGG_LABEL_ENCODER_PATH
class FaceRecognitionService:
    # def __init__(self):
    #     try:
    #         with open(FACE_ENCODINGS_PATH, 'rb') as f:
    #             self.known_faces = pickle.load(f)
    #     except FileNotFoundError:
    #         self.known_faces = {}
    #
    # def recognize(self, image):
    #     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     encodings = face_recognition.face_encodings(rgb_image)
    #     if not encodings:
    #         return "Unknown"
    #     for name, known_encoding in self.known_faces.items():
    #         if True in face_recognition.compare_faces([known_encoding], encodings[0]):
    #             return name
    #     return "Unknown"
    #
    # def add_face(self, name, images):
    #     encodings = []
    #     for image in images:
    #         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         enc = face_recognition.face_encodings(rgb_image)
    #         if enc:
    #             encodings.append(enc[0])
    #     if encodings:
    #         avg_encoding = np.mean(encodings, axis=0)
    #         self.known_faces[name] = avg_encoding
    #         with open(FACE_ENCODINGS_PATH, 'wb') as f:
    #             pickle.dump(self.known_faces, f)
    #         return True
    #     return False
    def __init__(self):
        # Tải mô hình SVM và encoder
        try:
            with open(VGG_SVM_MODEL_PATH, 'rb') as f:
                self.svm = pickle.load(f)
            with open(VGG_LABEL_ENCODER_PATH, 'rb') as f:
                self.encoder = pickle.load(f)
        except FileNotFoundError:
            self.svm = SVC(kernel="linear", probability=True)
            self.encoder = LabelEncoder()
            print("Warning: Model files not found. Initialized empty SVM and encoder.")

    def recognize(self, image):
        try:
            # Trích xuất embedding VGG-Face
            embedding = \
            DeepFace.represent(image, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)[0][
                "embedding"]

            # Dự đoán với SVM
            pred = self.svm.predict([embedding])[0]
            student_name = self.encoder.inverse_transform([pred])[0]
            return student_name
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown"

    def add_face(self, name, images):
        try:
            embeddings = []
            for image in images:
                # Trích xuất embedding cho mỗi ảnh
                embedding = \
                DeepFace.represent(image, model_name="VGG-Face", detector_backend="opencv", enforce_detection=False)[0][
                    "embedding"]
                embeddings.append(embedding)

            if not embeddings:
                print("No valid embeddings generated")
                return False

            # Tính embedding trung bình (tùy chọn, hoặc dùng tất cả embeddings để huấn luyện lại)
            avg_embedding = np.mean(embeddings, axis=0)

            # Lấy danh sách embeddings và labels hiện tại
            try:
                with open("embeddings_cache.pkl", "rb") as f:
                    cache = pickle.load(f)
                existing_embeddings = cache["embeddings"]
                existing_labels = cache["labels"]
            except FileNotFoundError:
                existing_embeddings = []
                existing_labels = []

            # Thêm embedding và label mới
            existing_embeddings.append(avg_embedding)
            existing_labels.append(name)

            # Cập nhật encoder
            self.encoder.fit(existing_labels)
            labels_encoded = self.encoder.transform(existing_labels)

            # Huấn luyện lại SVM
            self.svm.fit(existing_embeddings, labels_encoded)

            # Lưu model và encoder
            with open(VGG_SVM_MODEL_PATH, "wb") as f:
                pickle.dump(self.svm, f)
            with open(VGG_LABEL_ENCODER_PATH, "wb") as f:
                pickle.dump(self.encoder, f)

            # Lưu cache embeddings
            with open("embeddings_cache.pkl", "wb") as f:
                pickle.dump({"embeddings": existing_embeddings, "labels": existing_labels}, f)

            print(f"Added {name} with {len(embeddings)} images")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            return False