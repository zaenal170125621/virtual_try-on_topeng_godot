from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import base64
import numpy as np
import joblib
from train_dataset import extract_features
import uvicorn
import os
from typing import Optional

app = FastAPI(title="VTO Topeng Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model hasil training
model = joblib.load("models/face_pose_regressor.joblib")
scaler = model.get("scaler")

# Global state untuk mask yang dipilih
current_mask_path = "godot_project_virtual-try-on/assets/filter_image/pig_nose.png"
available_masks = {
    "pig_nose": "godot_project_virtual-try-on/assets/filter_image/pig_nose.png",
    # Tambahkan mask lain di sini
}

# Camera capture
cap = None


class MaskRequest(BaseModel):
    mask_name: str


@app.post("/select_mask")
def select_mask(req: MaskRequest):
    """Endpoint untuk memilih mask yang akan digunakan"""
    global current_mask_path
    if req.mask_name in available_masks:
        current_mask_path = available_masks[req.mask_name]
        return {"status": "success", "mask": req.mask_name}
    return {"status": "error", "message": "Mask tidak ditemukan"}


@app.get("/available_masks")
def get_available_masks():
    """Endpoint untuk mendapatkan list mask yang tersedia"""
    return {"masks": list(available_masks.keys())}


def apply_mask_to_face(frame, mask_img, bbox, roll, scale):
    """Aplikasikan mask ke wajah dengan pose yang sesuai"""
    x, y, w, h = bbox

    # Adjust scale (model output bisa perlu tuning)
    # Scale 1.0 = ukuran wajah, kita buat mask sedikit lebih besar
    adjusted_scale = max(0.8, min(1.5, scale * 1.2))

    # Resize mask sesuai ukuran wajah dengan scale factor
    mask_w = int(w * adjusted_scale)
    mask_h = int(h * adjusted_scale)

    if mask_w <= 0 or mask_h <= 0:
        return frame

    mask_resized = cv2.resize(
        mask_img, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

    # PENTING: Inverse roll karena kamera adalah mirror image
    # Ketika kepala miring ke kanan, nilai roll positif, mask harus miring ke kiri
    adjusted_roll = -roll  # Inverse untuk mirror effect

    # Rotate mask sesuai roll dengan padding yang cukup
    # Hitung ukuran canvas yang cukup untuk rotasi
    diagonal = int(np.sqrt(mask_w**2 + mask_h**2))
    padded_size = diagonal + 20

    # Buat canvas dengan padding
    if mask_resized.shape[2] == 4:
        canvas = np.zeros((padded_size, padded_size, 4), dtype=np.uint8)
    else:
        canvas = np.zeros((padded_size, padded_size, 3), dtype=np.uint8)

    # Paste mask di tengah canvas
    paste_x = (padded_size - mask_w) // 2
    paste_y = (padded_size - mask_h) // 2
    canvas[paste_y:paste_y+mask_h, paste_x:paste_x+mask_w] = mask_resized

    # Rotate dengan center di tengah canvas
    center = (padded_size // 2, padded_size // 2)
    M = cv2.getRotationMatrix2D(center, adjusted_roll, 1.0)

    if canvas.shape[2] == 4:
        mask_rotated = cv2.warpAffine(canvas, M, (padded_size, padded_size),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0, 0))
    else:
        mask_rotated = cv2.warpAffine(canvas, M, (padded_size, padded_size),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))

    # Crop back to original size after rotation
    crop_x = (padded_size - mask_w) // 2
    crop_y = (padded_size - mask_h) // 2
    mask_rotated = mask_rotated[crop_y:crop_y+mask_h, crop_x:crop_x+mask_w]

    # Posisikan mask di tengah wajah
    mask_x = x + (w - mask_w) // 2
    mask_y = y + (h - mask_h) // 2

    # Pastikan mask tidak keluar dari frame
    src_x_start = 0
    src_y_start = 0
    src_x_end = mask_w
    src_y_end = mask_h

    if mask_x < 0:
        src_x_start = -mask_x
        mask_w = mask_w + mask_x
        mask_x = 0
    if mask_y < 0:
        src_y_start = -mask_y
        mask_h = mask_h + mask_y
        mask_y = 0
    if mask_x + mask_w > frame.shape[1]:
        src_x_end = src_x_start + (frame.shape[1] - mask_x)
        mask_w = frame.shape[1] - mask_x
    if mask_y + mask_h > frame.shape[0]:
        src_y_end = src_y_start + (frame.shape[0] - mask_y)
        mask_h = frame.shape[0] - mask_y

    if mask_w <= 0 or mask_h <= 0:
        return frame

    # Crop mask yang akan di-overlay
    mask_to_overlay = mask_rotated[src_y_start:src_y_end,
                                   src_x_start:src_x_end]

    if mask_to_overlay.shape[0] == 0 or mask_to_overlay.shape[1] == 0:
        return frame

    # Overlay mask dengan alpha blending
    if mask_to_overlay.shape[2] == 4:  # Has alpha channel
        alpha = mask_to_overlay[:, :, 3:4] / 255.0
        mask_rgb = mask_to_overlay[:, :, :3]

        roi = frame[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]
        blended = (alpha * mask_rgb + (1 - alpha) * roi).astype(np.uint8)
        frame[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = blended
    else:
        frame[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = mask_to_overlay

    return frame


def generate_frames():
    """Generator untuk streaming video dengan mask overlay"""
    global cap, current_mask_path

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Smoothing untuk stabilitas
    prev_roll = 0
    prev_scale = 1.0
    alpha = 0.7  # Smoothing factor (0 = no smoothing, 1 = max smoothing)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Mirror frame untuk user experience yang lebih natural
        frame = cv2.flip(frame, 1)

        # Load mask
        if os.path.exists(current_mask_path):
            mask_img = cv2.imread(current_mask_path, cv2.IMREAD_UNCHANGED)
        else:
            mask_img = None

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))

        if len(faces) > 0 and mask_img is not None:
            x, y, w, h = faces[0]
            roi = gray[y:y+h, x:x+w]

            # Predict pose
            try:
                feat = extract_features(roi).reshape(1, -1)
                if scaler:
                    feat = scaler.transform(feat)

                roll_pred = model["roll"].predict(feat)[0]
                scale_pred = model["scale"].predict(feat)[0]

                # Apply smoothing untuk mengurangi jitter
                roll = alpha * prev_roll + (1 - alpha) * roll_pred
                scale = alpha * prev_scale + (1 - alpha) * scale_pred

                prev_roll = roll
                prev_scale = scale

                # Apply mask
                frame = apply_mask_to_face(
                    frame, mask_img, (x, y, w, h), roll, scale)

                # Debug info (optional - uncomment untuk debugging)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # cv2.putText(frame, f"Roll: {roll:.1f}", (x, y-30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.putText(frame, f"Scale: {scale:.2f}", (x, y-10),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error applying mask: {e}")
                import traceback
                traceback.print_exc()

        # Encode frame dengan quality tinggi
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video_feed")
def video_feed():
    """Endpoint untuk streaming video dengan mask overlay"""
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/get_frame")
def get_frame():
    """Endpoint untuk mendapatkan single frame (lebih stabil untuk Godot)"""
    global cap, current_mask_path

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    success, frame = cap.read()
    if not success:
        from fastapi.responses import Response
        return Response(status_code=503, content="Camera not available")

    # Load mask
    if os.path.exists(current_mask_path):
        mask_img = cv2.imread(current_mask_path, cv2.IMREAD_UNCHANGED)
    else:
        mask_img = None

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0 and mask_img is not None:
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]

        # Predict pose
        try:
            feat = extract_features(roi).reshape(1, -1)
            if scaler:
                feat = scaler.transform(feat)
            roll = model["roll"].predict(feat)[0]
            scale = model["scale"].predict(feat)[0]

            # Apply mask
            frame = apply_mask_to_face(
                frame, mask_img, (x, y, w, h), roll, scale)
        except Exception as e:
            print(f"Error applying mask: {e}")

    # Encode frame to JPEG with high quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ret, buffer = cv2.imencode('.jpg', frame, encode_param)

    if not ret:
        from fastapi.responses import Response
        return Response(status_code=500, content="Failed to encode frame")

    from fastapi.responses import Response
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.on_event("shutdown")
def shutdown_event():
    """Release camera saat shutdown"""
    global cap
    if cap is not None:
        cap.release()


# Jalankan server lokal
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
