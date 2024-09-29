import os
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('SDAI.h5')

def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')
    
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = preprocess_image(img_path)

    prediction = model.predict(img)
    prediction_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    app.run(debug=True)
