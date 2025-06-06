import joblib

def check_file(file_path):
    try:
        data = joblib.load(file_path)
        print(f"File {file_path} loaded successfully: {type(data)}")
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return False

# Kiểm tra cả hai file
scaler_path = "./models/scaler.pkl"
svm_path = "./models/face_recognition_svm.pkl"

check_file(scaler_path)
check_file(svm_path)