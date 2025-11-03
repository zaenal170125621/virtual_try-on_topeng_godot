# Scripts/webcam_server.py
from flask import Flask, Response
import cv2
import numpy as np
import mediapipe as mp
from math import atan2, degrees, hypot

# ==== Flask Setup ====
app = Flask(__name__)

# ==== Kamera ====
camera = cv2.VideoCapture(0)

# ==== Mediapipe Face Mesh ====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==== Fungsi bantu ====
def get_angle(x1, y1, x2, y2):
    """Hitung sudut antara dua titik dalam derajat."""
    return degrees(atan2(y2 - y1, x2 - x1))

# ==== Gambar overlay ====
nose_image = cv2.imread("C:/1 Semester 5 Tugas Pemrograman/godot-computer-vision/virtual-try-on/assets/filter_image/pig_nose.png", cv2.IMREAD_UNCHANGED)
if nose_image is None:
    raise FileNotFoundError("Gagal memuat pig_nose.png, periksa path-nya.")

# ==== Frame generator untuk streaming ====
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Konversi ke RGB untuk Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Landmark hidung
                nose_tip = face_landmarks.landmark[1]
                nose_left = face_landmarks.landmark[49]
                nose_right = face_landmarks.landmark[279]

                # Konversi ke koordinat piksel
                cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
                lx, ly = int(nose_left.x * w), int(nose_left.y * h)
                rx, ry = int(nose_right.x * w), int(nose_right.y * h)

                # Ukuran dan sudut hidung
                nose_width = int(hypot(lx - rx, ly - ry) * 1.8)
                nose_height = int(nose_width * 0.77)
                angle_degree = -get_angle(lx, ly, rx, ry)

                # Resize dan rotasi gambar hidung
                nose_resized = cv2.resize(nose_image, (nose_width, nose_height))
                center = (nose_width // 2, nose_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
                nose_rotated = cv2.warpAffine(
                    nose_resized,
                    rotation_matrix,
                    (nose_width, nose_height)
                )

                # Alpha blending
                if nose_rotated.shape[2] == 4:
                    alpha = nose_rotated[:, :, 3] / 255.0
                    pig_rgb = nose_rotated[:, :, :3]
                else:
                    pig_rgb = nose_rotated
                    alpha = np.ones((nose_height, nose_width), dtype=float)

                top_left = (int(cx - nose_width / 2), int(cy - nose_height / 2))
                x1, y1 = max(top_left[0], 0), max(top_left[1], 0)
                x2, y2 = min(x1 + nose_width, w), min(y1 + nose_height, h)

                roi = frame[y1:y2, x1:x2]
                h_roi, w_roi, _ = roi.shape

                if h_roi > 0 and w_roi > 0:
                    pig_crop = pig_rgb[0:h_roi, 0:w_roi]
                    alpha_crop = alpha[0:h_roi, 0:w_roi, None]
                    blended = (alpha_crop * pig_crop + (1 - alpha_crop) * roi).astype(np.uint8)
                    frame[y1:y2, x1:x2] = blended

        # Encode dan kirim ke browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Endpoint streaming video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Tes endpoint sederhana."""
    return "<h3>Webcam Server Aktif!<br>Stream di <a href='/video_feed'>/video_feed</a></h3>"

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)
