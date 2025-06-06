import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from config import MODEL_PATH, HAAR_CASCADE_PATH
import logging

class EmotionDetectionService:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            # Cập nhật đường dẫn mô hình
            self.model = load_model('/Users/nguyenquangtruong/Desktop/HocTap2/PBL5/SV_SmartClassroom/models/best_CNN6Model.keras')
            self.logger.info("Loaded emotion model: best_CNN6Model.keras")
            self.logger.info(f"Model input shape: {self.model.input_shape}")
            self.logger.info(f"Model output shape: {self.model.output_shape}")
            # Kiểm tra số chiều sau Flatten
            for layer in self.model.layers:
                if layer.name.startswith('flatten'):
                    self.logger.info(f"Flatten output shape: {layer.output.shape}")
            # Kiểm tra output shape phù hợp với 6 cảm xúc
            if self.model.output_shape[-1] != 6:
                raise ValueError(f"Model output shape {self.model.output_shape} does not match 6 emotions")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
            self.logger.error("Failed to load Haar Cascade")
            raise ValueError("Invalid Haar Cascade file")

        # Thứ tự cảm xúc của mô hình và đầu ra
        self.model_emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.output_emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def detect_faces_and_emotions(self, image):
        try:
            self.logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            self.logger.info(f"Detected {len(faces)} faces")

            results = []
            for (x, y, w, h) in faces:
                face_roi = image[y:y + h, x:x + w]
                # Giữ RGB, resize thành 100x100
                face = cv2.resize(face_roi, (100, 100))
                # Chuẩn hóa bằng resnet50.preprocess_input
                face = tf.keras.applications.resnet50.preprocess_input(face.astype(np.float32))
                face = np.expand_dims(face, axis=0)  # Shape: (1, 100, 100, 3)
                self.logger.info(f"Face input shape: {face.shape}")

                try:
                    # Dự đoán cảm xúc
                    prediction = self.model.predict(face, verbose=0)
                    self.logger.info(f"Prediction shape: {prediction.shape}")
                    total = float(sum(prediction[0]))
                    # Tính tỷ lệ phần trăm theo model_emotions
                    emotion_percentages = {
                        emotion: float((prob / total) * 100) if total > 0 else 0.0
                        for emotion, prob in zip(self.model_emotions, prediction[0])
                    }
                    # Ánh xạ sang output_emotions
                    emotion_percentages = {
                        emotion: emotion_percentages.get(emotion, 0.0)
                        for emotion in self.output_emotions
                    }
                except Exception as e:
                    self.logger.error(f"Error predicting emotions for face at ({x}, {y}, {w}, {h}): {str(e)}")
                    # Trả về cảm xúc mặc định (0%)
                    emotion_percentages = {emotion: 0.0 for emotion in self.output_emotions}

                results.append({
                    "face": face_roi,
                    "emotion_percentages": emotion_percentages
                })

            return results
        except Exception as e:
            self.logger.error(f"Error in detect_faces_and_emotions: {str(e)}")
            return []