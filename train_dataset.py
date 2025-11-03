# ==============================
#  train_dataset.py (versi enhanced)
#  Multi-feature + Random Forest untuk Virtual Try-On Topeng
# ==============================

import cv2
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import os
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------------------------------
# Ekstraksi fitur lengkap
# -------------------------------
def extract_features(img):
    """Extract multiple features: ORB, HOG, texture, and geometry"""
    if img is None or img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
        return np.zeros(128)
    
    features = []
    
    # 1. ORB features (32 dims)
    orb = cv2.ORB_create(500)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(des) == 0:
        orb_feat = np.zeros(32)
    else:
        orb_feat = np.mean(des, axis=0)
    features.append(orb_feat)
    
    # 2. HOG features (simplified - 36 dims)
    img_resized = cv2.resize(img, (64, 64))
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_feat = hog.compute(img_resized)
    if hog_feat is not None:
        hog_feat = hog_feat.flatten()[:36]  # Take first 36 dims
    else:
        hog_feat = np.zeros(36)
    features.append(hog_feat)
    
    # 3. Texture features (20 dims)
    # Aspect ratio
    aspect = img.shape[1] / max(img.shape[0], 1)
    # Mean and std intensity
    mean_val = np.mean(img)
    std_val = np.std(img)
    # Sobel edges
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_mean = np.mean(np.sqrt(sobelx**2 + sobely**2))
    # Histogram features
    hist = cv2.calcHist([img], [0], None, [16], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-7)
    
    texture_feat = np.concatenate([
        [aspect, mean_val, std_val, edge_mean],
        hist
    ])
    features.append(texture_feat)
    
    # 4. Geometric features from face landmarks (40 dims)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
    results = face_mesh.process(img_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        # Extract key points
        h, w = img.shape[:2]
        points = []
        key_indices = [
            0, 10, 152, 234, 454,  # Face outline
            33, 263, 61, 291,       # Eyes
            1, 4, 5, 195           # Nose and mouth
        ]
        for idx in key_indices:
            lm = landmarks.landmark[idx]
            points.append([lm.x * w, lm.y * h])
        points = np.array(points)
        
        # Calculate geometric features
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        # Combine features
        geom_feat = np.concatenate([
            distances,
            angles,
            [np.std(distances), np.std(angles)]
        ])
        
        # Pad or truncate to 40 dims
        if len(geom_feat) < 40:
            geom_feat = np.pad(geom_feat, (0, 40 - len(geom_feat)))
        else:
            geom_feat = geom_feat[:40]
    else:
        geom_feat = np.zeros(40)
    
    features.append(geom_feat)
    
    # Concatenate all features
    all_features = np.concatenate(features)
    return all_features

# -------------------------------
# Fungsi load dataset dengan augmentasi
# -------------------------------
def calculate_actual_pose(landmarks, img_shape):
    """Calculate actual roll and scale from face landmarks"""
    h, w = img_shape[:2]
    
    # Get key points
    points = []
    for lm in landmarks.landmark:
        points.append([lm.x * w, lm.y * h, lm.z])
    points = np.array(points)
    
    # Calculate roll (head tilt)
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    dx = (right_eye.x - left_eye.x) * w
    dy = (right_eye.y - left_eye.y) * h
    roll = np.degrees(np.arctan2(dy, dx))
    
    # Calculate scale (face size relative to image)
    face_points = points[:, :2]  # Only x, y
    bbox = [
        np.min(face_points[:, 0]),
        np.min(face_points[:, 1]),
        np.max(face_points[:, 0]),
        np.max(face_points[:, 1])
    ]
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    scale = max(face_width / w, face_height / h)
    
    return roll, max(scale, 0.1)  # Minimum scale 0.1

def augment_image(img):
    """Apply random augmentation to create variations"""
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img_aug = cv2.warpAffine(img, M, (w, h))
    
    # Random brightness
    brightness = np.random.uniform(0.7, 1.3)
    img_aug = np.clip(img_aug * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast
    contrast = np.random.uniform(0.8, 1.2)
    img_aug = np.clip((img_aug - 128) * contrast + 128, 0, 255).astype(np.uint8)
    
    return img_aug, angle

def load_data(folder, augment=False):
    X, y_roll, y_scale = [], [], []
    skipped = 0

    json_path = os.path.join(folder, "labels.json")
    if not os.path.exists(json_path):
        print(f"⚠️  File {json_path} tidak ditemukan!")
        return np.array(X), np.array(y_roll), np.array(y_scale)

    with open(json_path) as f:
        labels = json.load(f)

    print(f"📂 Loading from {folder}...")
    
    for item in tqdm(labels, desc=f"Processing", ncols=90):
        img_path = os.path.join(folder, "images", item["file"])
        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        # Ambil bounding box dan validasi
        try:
            x, y, w, h = item["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
        except Exception:
            skipped += 1
            continue

        # Perbaiki kalau keluar batas
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > img.shape[1]:
            w = img.shape[1] - x
        if y + h > img.shape[0]:
            h = img.shape[0] - y
        if w <= 5 or h <= 5:
            skipped += 1
            continue

        roi = img[y:y+h, x:x+w]
        if roi is None or roi.size == 0:
            skipped += 1
            continue
        
        # Calculate actual pose from face landmarks
        results = face_mesh.process(roi)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            actual_roll, actual_scale = calculate_actual_pose(landmarks, roi.shape)
        else:
            # Use default values if no face detected
            actual_roll = 0.0
            actual_scale = 1.0
        
        # Convert to grayscale for feature extraction
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Original sample
        feat = extract_features(roi_gray)
        X.append(feat)
        y_roll.append(actual_roll)
        y_scale.append(actual_scale)
        
        # Add augmented samples if requested
        if augment and np.random.random() < 0.3:  # 30% augmentation rate
            for _ in range(2):  # Add 2 augmented versions
                aug_img, aug_angle = augment_image(roi_gray)
                feat_aug = extract_features(aug_img)
                X.append(feat_aug)
                y_roll.append(actual_roll + aug_angle)  # Adjust roll by augmentation angle
                y_scale.append(actual_scale)

    print(f"✅ Loaded {len(X)} samples from {folder} ({skipped} skipped)")
    return np.array(X), np.array(y_roll), np.array(y_scale)

# -------------------------------
# Pipeline utama
# -------------------------------
def main():
    print("🚀 Memuat dataset...")
    X_train, y_roll_train, y_scale_train = load_data("dataset/train", augment=True)
    X_val, y_roll_val, y_scale_val = load_data("dataset/val", augment=False)

    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Dataset kosong atau path salah.")
        return

    print(f"\n📊 Jumlah data: {len(X_train)} train, {len(X_val)} val")
    
    # Check data variability
    print(f"📊 Roll range: [{y_roll_train.min():.2f}, {y_roll_train.max():.2f}] degrees")
    print(f"📊 Scale range: [{y_scale_train.min():.3f}, {y_scale_train.max():.3f}]")
    print(f"📊 Roll std: {y_roll_train.std():.2f}, Scale std: {y_scale_train.std():.3f}")
    
    # Normalize features
    print("\n⚙️  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("⚙️  Training model dengan Random Forest... (tunggu beberapa menit)\n")

    # Random Forest models (better than SVM for this task)
    rf_roll = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_scale = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("\n🔄 Training Roll predictor...")
    rf_roll.fit(X_train_scaled, y_roll_train)
    
    print("\n🔄 Training Scale predictor...")
    rf_scale.fit(X_train_scaled, y_scale_train)

    # Evaluasi
    print("\n📈 Evaluating model...")
    pred_roll = rf_roll.predict(X_val_scaled)
    pred_scale = rf_scale.predict(X_val_scaled)

    mae_roll = mean_absolute_error(y_roll_val, pred_roll)
    mae_scale = mean_absolute_error(y_scale_val, pred_scale)
    
    r2_roll = r2_score(y_roll_val, pred_roll)
    r2_scale = r2_score(y_scale_val, pred_scale)

    print("\n" + "="*60)
    print("✅ Training selesai!")
    print("="*60)
    print(f"📈 Validation MAE Roll : {mae_roll:.4f} degrees")
    print(f"📈 Validation MAE Scale: {mae_scale:.4f}")
    print(f"📈 Validation R² Roll  : {r2_roll:.4f}")
    print(f"📈 Validation R² Scale : {r2_scale:.4f}")
    print("="*60)
    
    # Show prediction examples
    print("\n🔍 Sample predictions (first 5):")
    print("Roll - Actual vs Predicted:")
    for i in range(min(5, len(y_roll_val))):
        print(f"  {y_roll_val[i]:7.2f}° → {pred_roll[i]:7.2f}° (error: {abs(y_roll_val[i]-pred_roll[i]):.2f}°)")
    print("\nScale - Actual vs Predicted:")
    for i in range(min(5, len(y_scale_val))):
        print(f"  {y_scale_val[i]:.3f} → {pred_scale[i]:.3f} (error: {abs(y_scale_val[i]-pred_scale[i]):.3f})")

    # Feature importance
    print("\n🔝 Top 10 important features:")
    importances = (rf_roll.feature_importances_ + rf_scale.feature_importances_) / 2
    top_indices = np.argsort(importances)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {importances[idx]:.4f}")

    # Simpan model
    os.makedirs("models", exist_ok=True)
    model_data = {
        "roll": rf_roll,
        "scale": rf_scale,
        "scaler": scaler,
        "feature_size": X_train.shape[1],
        "performance": {
            "mae_roll": float(mae_roll),
            "mae_scale": float(mae_scale),
            "r2_roll": float(r2_roll),
            "r2_scale": float(r2_scale)
        }
    }
    
    joblib.dump(model_data, "models/face_pose_regressor.joblib")
    print("\n💾 Model tersimpan di: models/face_pose_regressor.joblib")
    print("✨ Model siap digunakan untuk inference!")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()
