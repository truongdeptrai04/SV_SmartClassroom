import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from keras.models import load_model
from config import MODEL_PATH
import logging
import time

class EmotionDetectionService:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        try:
            self.model = load_model(MODEL_PATH)
            self.logger.info(f"Loaded emotion model: {MODEL_PATH}")
            self.logger.info(f"Model input shape: {self.model.input_shape}")
            self.logger.info(f"Model output shape: {self.model.output_shape}")
            for layer in self.model.layers:
                if layer.name.startswith('flatten'):
                    self.logger.info(f"Flatten output shape: {layer.output.shape}")
            if self.model.output_shape[-1] != 6:
                raise ValueError(f"Model output shape {self.model.output_shape} does not match 6 emotions")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

        self.model_emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        self.output_emotions = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def detect_faces(self, image):
        try:
            self.logger.info(f"Starting face detection, input shape: {image.shape}, dtype: {image.dtype}")
            start_time = time.time()
            face_results = DeepFace.analyze(
                image,
                actions=['emotion'],
                detector_backend='retinaface',
                enforce_detection=False
            )
            detect_time = time.time() - start_time
            self.logger.info(f"Completed face detection, detected {len(face_results)} faces (took {detect_time:.2f}s)")
            results = []
            for result in face_results:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                if w <= 0 or h <= 0:
                    self.logger.warning(f"Invalid face dimensions at ({x}, {y}, {w}, {h})")
                    continue
                face_roi = image[y:y + h, x:x + w]
                results.append({
                    "face": face_roi,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })
            return results
        except Exception as e:
            self.logger.error(f"Error in detect_faces: {str(e)}")
            return []

    def detect_emotions(self, face_roi):
        try:
            self.logger.info("Starting emotion detection")
            start_time = time.time()
            face = cv2.resize(face_roi, (100, 100))
            face = tf.keras.applications.resnet50.preprocess_input(face.astype(np.float32))
            face = np.expand_dims(face, axis=0)
            self.logger.info(f"Face input shape: {face.shape}")

            prediction = self.model.predict(face, verbose=0)
            self.logger.info(f"Prediction shape: {prediction.shape}")
            total = float(sum(prediction[0]))
            emotion_percentages = {
                emotion: float((prob / total) * 100) if total > 0 else 0.0
                for emotion, prob in zip(self.model_emotions, prediction[0])
            }
            emotion_percentages = {
                emotion: emotion_percentages.get(emotion, 0.0)
                for emotion in self.output_emotions
            }
            detect_time = time.time() - start_time
            self.logger.info(f"Completed emotion detection (took {detect_time:.2f}s)")
            return emotion_percentages
        except Exception as e:
            self.logger.error(f"Error in detect_emotions: {str(e)}")
            return {emotion: 0.0 for emotion in self.output_emotions}