# Virtual Try-On Topeng dengan Godot

Proyek ini adalah aplikasi virtual try-on topeng yang menggunakan Godot Engine sebagai frontend dan backend Python dengan FastAPI untuk deteksi wajah dan overlay topeng secara real-time. Sistem ini menggunakan model machine learning untuk memperkirakan pose wajah (roll dan scale) agar topeng dapat menyesuaikan dengan gerakan kepala.

## Fitur Utama

- **Deteksi Wajah Real-Time**: Menggunakan Haar Cascade untuk mendeteksi wajah dari webcam.
- **Overlay Topeng Dinamis**: Topeng menyesuaikan rotasi dan skala berdasarkan pose wajah.
- **Multi-Mask Support**: Mendukung berbagai topeng dengan dropdown untuk pemilihan.
- **Model Machine Learning**: Random Forest untuk regresi pose wajah (roll dan scale).
- **Integrasi Godot**: UI interaktif dengan Godot Engine menggunakan C#.
- **API Backend**: Endpoint REST untuk streaming video dan kontrol topeng.

## Struktur Proyek

```
virtual_try-on_topeng_godot/
├── app.py                          # Backend utama (FastAPI)
├── backend_app.py                  # Alternatif backend
├── backend_flask.py                # Backend Flask (alternatif)
├── train_dataset.py                # Script training model ML
├── start_backend.bat               # Script untuk menjalankan backend
├── combine_yolo_labels.py          # Script untuk menggabung label YOLO
├── godot_project_virtual-try-on/   # Proyek Godot
│   ├── scene/
│   │   ├── try_on.tscn            # Scene utama try-on
│   │   └── try_on_new.tscn        # Scene alternatif
│   └── scripts/
│       ├── TryOn.cs               # Script C# untuk try-on
│       └── TryOn.gd               # Script GDScript (alternatif)
├── models/                         # Model ML terlatih
│   └── face_pose_regressor.joblib
├── dataset/                        # Dataset (tidak di-push ke Git)
│   ├── labels/
│   ├── labels2/
│   ├── train/
│   └── val/
├── __pycache__/                    # Cache Python
├── CARA_TAMBAH_TOPENG.md           # Panduan menambah topeng
├── INTEGRATION_GUIDE.md            # Panduan integrasi
├── README_FIXES.md                 # Log perbaikan error
├── README_USAGE.md                 # Panduan penggunaan
└── README.md                       # File ini
```

## Persyaratan Sistem

- **Python 3.8+**
- **Godot Engine 4.x**
- **Webcam** untuk input video
- **Git** untuk version control

## Dependencies Python

Install dependencies dengan:

```bash
pip install fastapi uvicorn opencv-python scikit-learn joblib numpy
```

Atau gunakan virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt  # Jika ada
```

## Cara Menjalankan

### 1. Jalankan Backend

Gunakan script batch yang disediakan:

```bash
start_backend.bat
```

Atau jalankan manual:

```bash
python app.py
```

Backend akan berjalan di `http://localhost:8000`.

### 2. Jalankan Godot Project

1. Buka Godot Engine.
2. Import proyek dari folder `godot_project_virtual-try-on`.
3. Jalankan scene `try_on.tscn`.
4. Pilih topeng dari dropdown dan nikmati virtual try-on!

## Endpoint API

- `GET /video_feed`: Stream MJPEG untuk video dengan overlay topeng.
- `GET /get_frame`: Mendapatkan frame tunggal sebagai JPEG.
- `GET /available_masks`: List topeng yang tersedia.
- `POST /select_mask`: Pilih topeng untuk overlay.

## Training Model

Untuk melatih ulang model pose estimation:

```bash
python train_dataset.py
```

Model akan disimpan di `models/face_pose_regressor.joblib`.

## Menambah Topeng Baru

Lihat panduan di [`CARA_TAMBAH_TOPENG.md`](CARA_TAMBAH_TOPENG.md) untuk instruksi lengkap menambah topeng baru.

## Dokumentasi Tambahan

- [`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md): Panduan integrasi backend dengan Godot.
- [`README_FIXES.md`](README_FIXES.md): Log perbaikan bug dan error.
- [`README_USAGE.md`](README_USAGE.md): Panduan penggunaan aplikasi.

## Troubleshooting

- **Error JPEG parsing di Godot**: Pastikan backend menggunakan endpoint `/get_frame` untuk polling frame tunggal.
- **Topeng tidak berputar**: Periksa model pose estimation dan logika rotasi di `app.py`.
- **Webcam tidak terdeteksi**: Pastikan webcam terhubung dan tidak digunakan aplikasi lain.

## Kontribusi

1. Fork repository ini.
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`).
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`).
4. Push ke branch (`git push origin feature/AmazingFeature`).
5. Buat Pull Request.

## Lisensi

Proyek ini menggunakan lisensi MIT. Lihat file `LICENSE` untuk detail.

## Kontak

Jika ada pertanyaan atau masalah, buat issue di repository ini atau hubungi maintainer.

---

**Catatan**: Folder `dataset/` tidak di-push ke repository untuk menghemat space. Pastikan Anda memiliki dataset lokal untuk training model.
