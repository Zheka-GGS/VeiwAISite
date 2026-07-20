from flask import Flask, request, render_template_string, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'cifar10_classifier.h5'
model = None

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

HTML = '''
<!doctype html>
<html>
<head>
  <title>CIFAR-10 Classifier</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
    .box { border: 2px dashed #aaa; padding: 40px; text-align: center; }
    button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
    .result { margin-top: 20px; padding: 15px; background: #e0ffe0; display: none; }
  </style>
</head>
<body>
  <h1>CIFAR-10 Classifier</h1>
  <div class="box" onclick="document.getElementById('file').click()">
    <input type="file" id="file" accept="image/*" onchange="preview()">
    <p>Click to select image</p>
    <img id="img" style="max-width:200px; display:none;">
  </div>
  <button id="btn" disabled onclick="predict()">Predict</button>
  <div class="result" id="result"></div>

  <script>
    function preview() {
      const f = document.getElementById('file').files[0];
      if (!f) return;
      const r = new FileReader();
      r.onload = e => { document.getElementById('img').src = e.target.result; document.getElementById('img').style.display = 'block'; document.getElementById('btn').disabled = false; };
      r.readAsDataURL(f);
    }
    async function predict() {
      const f = document.getElementById('file').files[0];
      const data = new FormData();
      data.append('image', f);
      const res = await fetch('/predict', { method: 'POST', body: data });
      const j = await res.json();
      const el = document.getElementById('result');
      if (j.success) {
        el.innerHTML = 'Class: ' + j.class_name + '<br>Confidence: ' + (j.confidence * 100).toFixed(2) + '%';
        el.style.display = 'block';
      } else {
        el.innerHTML = 'Error: ' + j.error;
        el.style.background = '#ffe0e0';
        el.style.display = 'block';
      }
    }
  </script>
</body>
</html>
'''


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError('Model not found. Run train_cifar10.ipynb first.')
        model = load_model(MODEL_PATH)
    return model


def prepare_image(file):
    img = Image.open(file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((32, 32))
    arr = img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'no image'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'success': False, 'error': 'no file'}), 400

        m = get_model()
        img = prepare_image(f)
        preds = m.predict(img, verbose=0)
        idx = int(np.argmax(preds[0]))
        return jsonify({
            'success': True,
            'class_name': CLASS_NAMES[idx],
            'confidence': float(preds[0][idx])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print('Starting on http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
