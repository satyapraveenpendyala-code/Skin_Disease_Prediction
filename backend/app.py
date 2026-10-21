import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['USE_TF'] = '0'  # stops transformers from importing TensorFlow
import uuid
import numpy as np
import tensorflow as tf
import torch
import cv2
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from transformers import CLIPModel, CLIPProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_prediction.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Multimodal AI Model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Loading CLIP skin checker (first run downloads ~600MB)...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

CLASSES = {
    0: 'Actinic keratoses', 1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions', 3: 'Dermatofibroma',
    4: 'Melanoma', 5: 'Melanocytic nevi', 6: 'Vascular lesions'
}

GENDER_MAP = {'female': [1, 0, 0], 'male': [0, 1, 0], 'unknown': [0, 0, 1]}
# VERIFY: this order must be the same one used when the model was trained
LOC_MAP = {'back': 0, 'face': 1, 'chest': 2, 'upper extremity': 3,
           'lower extremity': 4, 'trunk': 5, 'abdomen': 6, 'scalp': 7}


def preprocess_metadata(age, gender, localization):
    age_scaled = (float(age) - 51.8) / 16.9
    gender_vec = GENDER_MAP.get(gender.lower(), [0, 0, 1])
    loc_vec = np.zeros(15)
    loc_vec[LOC_MAP.get(localization.lower(), 0)] = 1
    return np.hstack([[age_scaled], gender_vec, loc_vec]).astype('float32')


# --- CLIP gate: only close-up lesion images pass ---
# Each label has a rejection message. Only label 0 (lesion close-up) is accepted.
CLIP_LABELS = [
    ("a dermoscopy close-up image of a skin lesion, mole or spot", None),
    ("a close-up photo of plain healthy skin with no mole or spot",
     "No visible lesion found. Please upload a close-up of the mole or spot."),
    ("a photo of a person, face, selfie or group of people",
     "This looks like a photo of people. Please upload a close-up of the skin lesion only."),
    ("a photo of an object, shoe, clothing, animal, food, room, landscape or text",
     "This does not look like a skin image. Please upload a close-up of the skin lesion."),
]
LESION_THRESHOLD = 0.5  # raise to be stricter, lower if real lesions get rejected


SPOT_DARKNESS = 18    # how much darker than the nearby skin a spot must be (lower = more lenient)
SPOT_MIN_AREA = 0.004  # smallest spot size, as a fraction of the image


def has_visible_spot(filepath):
    """True if there is a spot clearly darker than the skin around it, near the centre."""
    img = cv2.resize(cv2.imread(filepath), (224, 224))
    L = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0], (0, 0), 3)
    background = cv2.GaussianBlur(L, (0, 0), 40)        # removes uneven lighting and shadows
    darker = cv2.subtract(background, L)
    mask = (darker > SPOT_DARKNESS).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] / mask.size
        cx, cy = centroids[i] / 224
        if area >= SPOT_MIN_AREA and 0.15 < cx < 0.85 and 0.15 < cy < 0.85:
            print(f"Spot found: {area * 100:.1f}% of image")
            return True
    print("No visible spot found")
    return False


def is_human_skin(filepath):
    try:
        img = Image.open(filepath).convert("RGB")
        texts = [t for t, _ in CLIP_LABELS]
        inputs = clip_proc(text=texts, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].tolist()
        print("CLIP:", {t[:25]: round(p, 2) for t, p in zip(texts, probs)})

        best = int(np.argmax(probs))
        if best == 0 and probs[0] >= LESION_THRESHOLD:
            # CLIP can't tell plain skin from a lesion, so also look for a real dark spot
            if not has_visible_spot(filepath):
                return False, CLIP_LABELS[1][1]
            return True, "Valid"
        msg = CLIP_LABELS[best][1] or "Image unclear. Please upload a sharper close-up of the lesion."
        return False, msg
    except Exception as e:
        print(f"Skin check error: {e}")
        return True, "Validation skipped"


def get_gradcam_heatmap(img_array, meta_array, model):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, meta_array])
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        predictions = tf.convert_to_tensor(predictions)
        conv_outputs = tf.convert_to_tensor(conv_outputs)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    age = request.form.get('age', 50)
    gender = request.form.get('gender', 'male')
    localization = request.form.get('localization', 'back')

    # safe, unique filename (no path tricks, no overwriting)
    ext = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(filepath)

    is_valid_skin, validation_message = is_human_skin(filepath)
    if not is_valid_skin:
        return jsonify({'diagnosis': 'Invalid Image', 'confidence': '0%',
                        'error_message': validation_message, 'heatmap_url': None})

    img = load_img(filepath, target_size=(224, 224))
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    meta_array = np.expand_dims(preprocess_metadata(age, gender, localization), axis=0)

    predictions = model.predict([img_array, meta_array], verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    diagnosis = CLASSES[pred_idx]
    confidence = float(predictions[0][pred_idx])

    heatmap_url = None

    if confidence < 0.70:
        # low confidence does NOT mean "no disease", so say it honestly
        diagnosis = f"Uncertain - closest match: {CLASSES[pred_idx]}"
    else:
        try:
            heatmap = get_gradcam_heatmap(img_array, meta_array, model)
            img_cv = cv2.imread(filepath)
            heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
            heatmap_filename = "heatmap_" + safe_name
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, heatmap_filename), superimposed_img)
            heatmap_url = f"http://127.0.0.1:5000/uploads/{heatmap_filename}"
        except Exception as e:
            print(f"Grad-CAM Error: {e}")

    return jsonify({
        'diagnosis': diagnosis,
        'confidence': f"{confidence * 100:.2f}%",
        'heatmap_url': heatmap_url
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)