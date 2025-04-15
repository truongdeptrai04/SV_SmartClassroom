import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import MODEL_PATH, HAAR_CASCADE_PATH


class EmotionDetectionService:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def detect_faces_and_emotions(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        results = []
        for (x, y, w, h) in faces:
            face_roi = image[y:y + h, x:x + w]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face_gray, (48, 48))
            face = np.expand_dims(face, axis=-1)
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = self.model.predict(face)[0]
            total = float(sum(prediction))  # Chuyển total sang float
            emotion_percentages = {emotion: float((prob / total) * 100) for emotion, prob in
                                   zip(self.emotions, prediction)}
            results.append({"face": face_roi, "emotion_percentages": emotion_percentages})

        return results