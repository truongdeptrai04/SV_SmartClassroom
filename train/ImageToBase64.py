from tensorflow.keras.models import load_model
model = load_model("models/fer2013_model.h5")
print(model.input_shape)  # Kết quả mong đợi: (None, 224, 224, 1) hoặc (None, 224, 224, 3)