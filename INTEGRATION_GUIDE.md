# Panduan Integrasi Virtual Try-On Topeng

## 📋 Persiapan

### 1. Install Dependencies
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install packages
pip install fastapi uvicorn opencv-python joblib scikit-learn numpy
```

### 2. Struktur File
```
virtual_try-on_topeng_godot/
├── app.py                          # Backend API (FastAPI)
├── train_dataset.py                # Training script & feature extraction
├── models/
│   └── face_pose_regressor.joblib  # Trained model
├── godot_project_virtual-try-on/
│   ├── assets/
│   │   └── filter_image/           # Folder untuk topeng/masks
│   │       ├── pig_nose.png
│   │       ├── mask1.png
│   │       └── mask2.png
│   └── scene/
│       └── try_on.tscn             # Scene dengan dropdown
```

## 🎭 Menambahkan Topeng Baru

### Langkah 1: Tambah File Gambar Topeng
1. Siapkan file PNG dengan transparansi (alpha channel)
2. Simpan di: `godot_project_virtual-try-on/assets/filter_image/`
3. Nama file contoh: `mask_batman.png`, `mask_spiderman.png`, dll.

### Langkah 2: Daftarkan di app.py
Edit `app.py`, cari bagian `available_masks`:
```python
available_masks = {
    "pig_nose": "godot_project_virtual-try-on/assets/filter_image/pig_nose.png",
    "batman": "godot_project_virtual-try-on/assets/filter_image/mask_batman.png",
    "spiderman": "godot_project_virtual-try-on/assets/filter_image/mask_spiderman.png",
    # Tambahkan mask lainnya di sini
}
```

## 🚀 Cara Menjalankan

### 1. Jalankan Backend
```bash
cd "d:\KULIAT\SEMESTER 5\PengolahanCitraDigital\virtual_try-on_topeng_godot"
.\.venv\Scripts\Activate.ps1
python app.py
```

Backend akan berjalan di: `http://localhost:5000`

### 2. Jalankan Godot Project
1. Buka Godot Engine
2. Import project: `godot_project_virtual-try-on`
3. Run scene: `try_on.tscn`
4. Pilih topeng dari dropdown
5. Kamera akan menampilkan topeng sesuai pose wajah

## 📡 API Endpoints

### 1. Get Available Masks
```http
GET http://localhost:5000/available_masks
```
Response:
```json
{
  "masks": ["pig_nose", "batman", "spiderman"]
}
```

### 2. Select Mask
```http
POST http://localhost:5000/select_mask
Content-Type: application/json

{
  "mask_name": "batman"
}
```

### 3. Video Stream
```http
GET http://localhost:5000/video_feed
```
Returns: MJPEG stream dengan mask overlay

## 🎯 Fitur

✅ **Multi-mask support** - Pilih berbagai topeng via dropdown  
✅ **Real-time pose detection** - Roll & scale prediction  
✅ **Face tracking** - Auto-detect wajah dengan Haar Cascade  
✅ **Alpha blending** - Transparansi topeng yang halus  
✅ **Auto-rotation** - Topeng mengikuti rotasi kepala  
✅ **Auto-scaling** - Ukuran topeng menyesuaikan wajah  

## 🔧 Troubleshooting

### Camera tidak muncul
- Pastikan webcam terkoneksi
- Cek permission camera di Windows Settings
- Restart app.py

### Mask tidak muncul
- Cek path file mask di `available_masks`
- Pastikan file PNG memiliki alpha channel
- Cek console untuk error messages

### Performance lambat
- Kurangi resolusi kamera di app.py (line 145-146)
- Reduce model complexity
- Close background applications

## 📊 Model Performance

- **Roll MAE**: 2.07° (Excellent)
- **Scale MAE**: 0.044 (Good)
- **Roll R²**: 0.73
- **Scale R²**: 0.25

Model sudah production-ready untuk virtual try-on application!
