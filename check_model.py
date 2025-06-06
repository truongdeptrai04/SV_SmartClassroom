from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from config import MODEL_PATH

def custom_input_layer(**config):
    config.pop('batch_shape', None)
    config['batch_input_shape'] = config.get('batch_input_shape', [None, 48, 48, 1])
    return InputLayer(**config)

try:
    model = load_model(MODEL_PATH, custom_objects={'InputLayer': custom_input_layer})
    print(f"Loaded model successfully: {MODEL_PATH}")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")