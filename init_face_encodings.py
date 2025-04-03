import os
import cv2
import face_recognition
import pickle
import numpy as np


DATASET_DIR = "team_data"
OUTPUT_FILE = "face_encodings.pkl"

def generate_encodings():
    known_faces = {}

    for student_name in os.listdir(DATASET_DIR):
        student_dir = os.path.join(DATASET_DIR, student_name)
        if not os.path.isdir(student_dir):
            continue

        encodings = []
        print(f"Processing {student_name}...")

        for img_file in os.listdir(student_dir):
            img_path = os.path.join(student_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {img_path}")
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb_image)
            if enc:
                encodings.append(enc[0])
            else:
                print(f"No face detected in {img_path}")

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            known_faces[student_name] = avg_encoding
            print(f"Added {student_name} with {len(encodings)} images")
        else:
            print(f"No valid encodings for {student_name}")

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(known_faces, f)
    print(f"Saved encodings to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_encodings()