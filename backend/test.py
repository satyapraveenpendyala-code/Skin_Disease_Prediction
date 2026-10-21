import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_disease_prediction.h5")
CSV_PATH = os.path.join(BASE_DIR, "dataset", "HAM10000_metadata.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "all_images")
N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 2000   # python test.py 2000

CLASSES = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
REVERSE_CLASSES = {v: k for k, v in CLASSES.items()}
GENDER_MAP = {'female': [1, 0, 0], 'male': [0, 1, 0], 'unknown': [0, 0, 1]}
LOC_MAP = {'back': 0, 'face': 1, 'chest': 2, 'upper extremity': 3,
           'lower extremity': 4, 'trunk': 5, 'abdomen': 6, 'scalp': 7}   # same as app.py


def preprocess_metadata(age, gender, localization):
    age_scaled = (float(age) - 51.8) / 16.9
    gender_vec = GENDER_MAP.get(str(gender).lower(), [0, 0, 1])
    loc_vec = np.zeros(15)
    loc_vec[LOC_MAP.get(str(localization).lower(), 0)] = 1
    return np.hstack([[age_scaled], gender_vec, loc_vec]).astype('float32')


if not os.path.isdir(IMAGE_DIR):
    sys.exit(f"Image folder not found: {IMAGE_DIR}\nPut all HAM10000 .jpg files in backend/dataset/all_images")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

df = pd.read_csv(CSV_PATH).dropna(subset=['age'])
if N_IMAGES < len(df):
    df = df.sample(n=N_IMAGES, random_state=42)

images, metas, labels = [], [], []
print(f"Loading {len(df)} images...")
for _, row in df.iterrows():
    path = os.path.join(IMAGE_DIR, f"{row['image_id']}.jpg")
    if not os.path.exists(path):
        continue
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)          # FIX: model expects RGB, OpenCV reads BGR
    images.append(cv2.resize(img, (224, 224)).astype('float32') / 255.0)
    metas.append(preprocess_metadata(row['age'], row['sex'], row['localization']))
    labels.append(REVERSE_CLASSES[row['dx']])

if not images:
    sys.exit("No images found. Check the file names in backend/dataset/all_images")

X, M, y_true = np.array(images), np.array(metas), np.array(labels)
print("Running predictions...")
y_prob = model.predict([X, M], batch_size=32, verbose=1)
y_pred = y_prob.argmax(axis=1)

names = list(CLASSES.values())
print("\n" + "=" * 55)
print(f"  Images tested      : {len(y_true)}")
print(f"  Accuracy           : {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"  Balanced accuracy  : {balanced_accuracy_score(y_true, y_pred) * 100:.2f}%")
print("=" * 55 + "\n")
print(classification_report(y_true, y_pred, labels=range(7), target_names=names, digits=2, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=range(7))
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=names, columns=names).to_string(), "\n")

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=names, yticklabels=names)
plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150); plt.close()

y_bin = label_binarize(y_true, classes=range(7))
plt.figure(figsize=(9, 7))
for i, name in enumerate(names):
    if y_bin[:, i].sum() > 0:
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (one-vs-rest)')
plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(os.path.join(BASE_DIR, 'roc_curve.png'), dpi=150); plt.close()

print("Saved backend/confusion_matrix.png and backend/roc_curve.png")