import joblib

FACE_ENCODINGS_PATH = "./models/face_recognition_svm.pkl"
try:
    with open(FACE_ENCODINGS_PATH, 'rb') as f:
        known_faces = joblib.load(f)
    print("Known faces:", list(known_faces.keys()))
except FileNotFoundError:
    print("File face_encodings.pkl không tồn tại")