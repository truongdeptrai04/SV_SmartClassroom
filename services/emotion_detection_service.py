import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import MODEL_PATH, HAAR_CASCADE_PATH

class EmotionDetectionService:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def detect(self, image):
        face = cv2.resize(image, (224, 224))
        face = np.stack([face] * 3, axis=-1) if len(image.shape) == 2 else face
        face = face/255.0
        face = np.expand_dims(face, axis=0)

        prediction = self.model.predict(face)[0]
        total = sum(prediction)
        emotion_percentages = {emotion: (prob / total) * 100 for emotion, prob in zip(self.emotions, prediction)}
        return emotion_percentages