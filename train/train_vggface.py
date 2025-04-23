import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pickle

# Đường dẫn đến dataset
dataset_path = "../team_data"
students = ["Truong", "Dan", "Trong", "Dung"]
embeddings = []
labels = []

# Trích xuất embeddings
for student in students:
    student_path = os.path.join(dataset_path, student)
    if not os.path.isdir(student_path):
        print(f"Directory not found: {student_path}")
        continue

    processed_count = 0
    for file in os.listdir(student_path):
        if file.endswith((".jpg", ".png")):
            image_path = os.path.join(student_path, file)
            # Kiểm tra ảnh có thể đọc
            img = cv2.imread(image_path)
            if img is None:
                print(f"Invalid image: {image_path}")
                continue
            try:
                # Trích xuất embedding với VGG-Face
                embedding = DeepFace.represent(image_path, model_name="VGG-Face", detector_backend="opencv",
                                               enforce_detection=False)
                embeddings.append(embedding[0]["embedding"])
                labels.append(student)
                processed_count += 1
                print(f"Processed {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"Processed {processed_count} images for {student}")

# Kiểm tra embeddings
if not embeddings:
    print("Error: No embeddings generated. Check dataset or DeepFace configuration.")
    exit(1)

# Chuyển labels thành số
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Huấn luyện SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(embeddings, labels_encoded)
print("SVM training completed")

# Lưu model và encoder
with open("../vggface_svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)
with open("../vggface_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(f"Model saved successfully. Total embeddings: {len(embeddings)}")