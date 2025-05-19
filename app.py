import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf  # tensorflow-cpu in requirements.txt

# Initialize Flask app
app = Flask(__name__)

# Vercel-compatible config (uses /tmp for storage)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='optimized_model.tflite')
interpreter.allocate_tensors()

# Prediction function
def predict_breast_cancer(image_path):
    # Preprocess image (adjust to your model's requirements)
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    pred = interpreter.get_tensor(output_details[0]['index'])
        # Map to YOUR original classes
    classes = ["normal", "sick", "unknown"]  # Must match your model's training labels
    predicted_class = classes[np.argmax(pred)]
    return {
        'prediction': ['normal', 'benign', 'malignant'][np.argmax(pred)],
        'confidence': float(np.max(pred))
    }

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction
        result = predict_breast_cancer(filepath)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check
@app.route('/')
def home():
    return "Breast Cancer Detection API"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))