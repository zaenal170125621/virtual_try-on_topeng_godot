# === train_dataset.py ===
import cv2, numpy as np, joblib, json
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

def extract_features(img):
    orb = cv2.ORB_create(500)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        return np.zeros(32)
    return np.mean(des, axis=0)

def load_data(folder):
    X, y_roll, y_scale = [], [], []
    with open(f"{folder}/labels.json") as f:
        labels = json.load(f)
    for item in labels:
        img_path = f"{folder}/images/{item['file']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x, y, w, h = item["bbox"]
        roi = img[y:y+h, x:x+w]
        X.append(extract_features(roi))
        y_roll.append(item["roll"])
        y_scale.append(item["scale"])
    return np.array(X), np.array(y_roll), np.array(y_scale)

# Load data
X_train, y_roll_train, y_scale_train = load_data("dataset/train")
X_val, y_roll_val, y_scale_val = load_data("dataset/val")

print(f"✅ Loaded {len(X_train)} train samples and {len(X_val)} val samples")

# Train dua model
svm_roll = SVR(kernel="rbf", C=10).fit(X_train, y_roll_train)
svm_scale = SVR(kernel="rbf", C=10).fit(X_train, y_scale_train)

# Evaluasi di validation
pred_roll = svm_roll.predict(X_val)
pred_scale = svm_scale.predict(X_val)

print("📊 MAE Roll :", mean_absolute_error(y_roll_val, pred_roll))
print("📊 MAE Scale:", mean_absolute_error(y_scale_val, pred_scale))

# Simpan model
joblib.dump({"roll": svm_roll, "scale": svm_scale}, "models/face_pose_regressor.joblib")
print("✅ Model tersimpan di models/face_pose_regressor.joblib")
