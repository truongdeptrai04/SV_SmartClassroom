import face_recognition
import pickle
import cv2
import numpy as np
from config import FACE_ENCODINGS_PATH

class FaceRecognitionService:
    def __init__(self):
        try:
            with open(FACE_ENCODINGS_PATH, 'rb') as f:
                self.known_faces = pickle.load(f)
        except FileNotFoundError:
            self.known_faces = {}

    def recognize(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if not encodings:
            return "Unknown"
        for name, known_encoding in self.known_faces.items():
            if True in face_recognition.compare_faces([known_encoding], encodings[0]):
                return name
        return "Unknown"

    def add_face(self, name, images):
        encodings = []
        for image in images:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb_image)
            if enc:
                encodings.append(enc[0])
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            self.known_faces[name] = avg_encoding
            with open(FACE_ENCODINGS_PATH, 'wb') as f:
                pickle.dump(self.known_faces, f)
            return True
        return False