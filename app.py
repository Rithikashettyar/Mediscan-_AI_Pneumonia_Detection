from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import os
from datetime import datetime, timedelta

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_AGE_DAYS = 7

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

MODEL_PATH = "pneumonia_model.h5"
model_loaded, model = False, None

try:
    model = load_model(MODEL_PATH, compile=False)
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load error: {e}")

CLASSES = ["Normal", "Pneumonia"]
INFO = {
    "Normal": {"status": "Healthy", "color": "success", "icon": "✓", "msg": "No pneumonia detected."},
    "Pneumonia": {"status": "Alert", "color": "danger", "icon": "⚠", "msg": "Pneumonia detected. Consult a doctor."}
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(img):
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0).astype(np.float32)

def clean_old_uploads():
    now = datetime.now()
    for f in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(path):
            file_time = datetime.fromtimestamp(os.path.getmtime(path))
            if now - file_time > timedelta(days=MAX_FILE_AGE_DAYS):
                os.remove(path)

@app.route('/')
def home():
    clean_old_uploads()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"success": False, "error": "Model not loaded."}), 500
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    try:
        filename = secure_filename(file.filename)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img_input = preprocess(img)
        preds = model.predict(img_input, verbose=0)
        idx = int(np.argmax(preds))
        label = CLASSES[idx]
        conf = round(float(np.max(preds)) * 100, 2)
        info = INFO[label]

        return jsonify({
            "success": True,
            "prediction": label,
            "status": info["status"],
            "confidence": conf,
            "color": info["color"],
            "icon": info["icon"],
            "message": info["msg"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "saved_path": filepath,
            "probabilities": {
                "Normal": round(float(preds[0][0]) * 100, 2),
                "Pneumonia": round(float(preds[0][1]) * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model_loaded})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080)) 
    print(f"MediScan AI running on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
