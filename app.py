from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

app = Flask(__name__)

# Upload folder config
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import os
import gdown

MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        # This is your shared Drive file ID
        file_id = "1i-uE2sN-Lh96WVe1AkegbyE8QQ0eoNLq"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded.")

# Call this before model load
download_model()

# Now you can safely load the model
from keras.models import load_model
model = load_model(MODEL_PATH)



# Class labels
class_names = {
    0: "Actinic keratosis",
    1: "Atopic Dermatitis",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic nevus",
    5: "Melanoma",
    6: "Squamous cell carcinoma",
    7: "Tinea Ringworm Candidiasis",
    8: "Vascular lesion"
}

# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/disease")
def disease():
    return render_template("diseases.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")
# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('prediction'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]

        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('prediction.html', prediction=pred_class, image_url=image_url)

    return "Something went wrong"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
