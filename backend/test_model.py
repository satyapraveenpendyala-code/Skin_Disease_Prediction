import os
import argparse
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --- 1. SETTINGS & PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "skin_disease_prediction.h5")
CSV_PATH = os.path.join(BASE_DIR, "backend", "dataset", "HAM10000_metadata.csv")

CLASSES = {
    0: "Actinic keratoses", 1: "Basal cell carcinoma", 2: "Benign keratosis-like lesions",
    3: "Dermatofibroma", 4: "Melanoma", 5: "Melanocytic nevi", 6: "Vascular lesions",
}

# --- 2. LOAD ENCODERS (must match training) ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Cannot find CSV at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df["age"] = df["age"].fillna(df["age"].mean())

sex_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[["sex"]])
loc_enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[["localization"]])
age_scaler = StandardScaler().fit(df[["age"]])

# --- 3. LOAD MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model inputs:", model.input_names)  # confirm order is [image, metadata]


def preprocess_input(img_path, age, sex, loc):
    # Image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # must match training
    img_array = np.expand_dims(img_array, axis=0)

    # Metadata (1 age + sex one-hot + localization one-hot = 19)
    age_scaled = age_scaler.transform(pd.DataFrame({"age": [age]}))
    sex_encoded = sex_enc.transform(pd.DataFrame({"sex": [sex]}))
    loc_encoded = loc_enc.transform(pd.DataFrame({"localization": [loc]}))
    meta_array = np.hstack([age_scaled, sex_encoded, loc_encoded]).astype("float32")

    return img_array, meta_array


def predict_image(image_path, age, sex, loc):
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    img_in, meta_in = preprocess_input(image_path, age, sex, loc)
    preds = model.predict([img_in, meta_in], verbose=0)[0]

    print("\n========================================")
    print(f"🖼️ Analyzing: {os.path.basename(image_path)}")
    print(f"Patient Data: Age {age}, Sex: {sex}, Loc: {loc}")
    print("========================================")

    for i, prob in enumerate(preds):
        print(f"{CLASSES[i].ljust(30)}: {prob * 100:05.2f}%")

    top = int(np.argmax(preds))
    print(f"\n🏆 PREDICTION: {CLASSES[top]} ({preds[top] * 100:.2f}%)\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Skin lesion prediction test")
    p.add_argument("image", help="Path to the lesion image")
    p.add_argument("--age", type=float, default=50)
    p.add_argument("--sex", default="unknown", choices=["male", "female", "unknown"])
    p.add_argument("--loc", default="unknown",
                   help="e.g. back, face, trunk, lower extremity, upper extremity, unknown")
    args = p.parse_args()

    predict_image(args.image, args.age, args.sex, args.loc)