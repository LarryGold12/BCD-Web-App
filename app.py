from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import os

# Try to use tflite-runtime (for PythonAnywhere), fallback to TensorFlow (for local dev)
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite-runtime (lightweight mode)")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("Using full TensorFlow (local mode)")

app = Flask(__name__)

# Class labels from your training
CLASS_NAMES = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
TARGET_SIZE = (224, 224)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="breast_cancer_model_xray.tflite")
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        img_path = 'temp_img.' + file.filename.rsplit('.', 1)[1].lower()
        file.save(img_path)

        try:
            # Preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(TARGET_SIZE)
            img_array = np.array(img, dtype=np.float32) / 255.0  # normalize
            input_data = np.expand_dims(img_array, axis=0)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = int(np.argmax(output_data[0]))
            confidence = float(np.max(output_data[0]))

            result = {
                'prediction': predicted_class,
                'class_name': CLASS_NAMES[predicted_class],
                'confidence': confidence,
                'all_predictions': {
                    'Benign': float(output_data[0][0]),
                    'Malignant': float(output_data[0][1]),
                    'Normal': float(output_data[0][2])
                }
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
