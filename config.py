import os

MODEL_PATH = './models/fer2013_model.h5'
HAAR_CASCADE_PATH = './models/haarcascade_frontalface_default.xml'
FIREBASE_CREDENTIALS = os.path.join(
    os.path.dirname(__file__),
    "smartclassroom-470a8-firebase-adminsdk-fbsvc-096f0aa342.json"
)
FACE_ENCODINGS_PATH = "face_encodings.pkl"
VGG_SVM_MODEL_PATH = "vggface_svm_model.pkl"
VGG_LABEL_ENCODER_PATH = "vggface_label_encoder.pkl"