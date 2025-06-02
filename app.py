import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf  # Use tensorflow-cpu in requirements.txt for lighter deployments

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder (Vercel/Heroku/Render compatible)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='optimized_model.tflite')
interpreter.allocate_tensors()

def predict_breast_cancer(image_path):
    """Predict breast cancer class from an image."""
    # Preprocess image (adjust to match your model's training)
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
    
    # Map to your model's classes
    classes = ["normal", "benign", "malignant"]  # Update to match your labels
    return {
        'prediction': classes[np.argmax(pred)],
        'confidence': float(np.max(pred))
    }

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
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

@app.route('/')
def home():
    """Health check endpoint."""
    return "Breast Cancer Detection API - Healthy"

# Local development (Gunicorn ignores this in production)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
