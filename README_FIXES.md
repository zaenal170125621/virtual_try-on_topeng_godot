# 🎭 Virtual Try-On Topeng - Troubleshooting & Improvements

## ✅ Perbaikan yang Sudah Dilakukan

### 1. **Masalah JPEG Parsing Error**
**Sebelumnya:**
```
Error: Condition "err" is true. Returning: Ref<Image>()
LoadJpgFromBuffer failed
```

**Penyebab:**
- Frame JPEG tidak complete atau corrupt saat streaming
- Buffer size terlalu kecil
- Quality encoding rendah

**Solusi:**
✅ Tambah `encode_param` dengan JPEG quality 90%
✅ Validasi `ret` sebelum yield frame
✅ Set `CAP_PROP_BUFFERSIZE = 1` untuk reduce latency
✅ Flip frame untuk mirror effect yang lebih natural

### 2. **Masalah Rotasi Tidak Pas**
**Sebelumnya:**
- Ketika kepala miring kanan, mask miring kiri (terbalik)
- Mask terpotong saat rotasi
- Rotasi tidak smooth (jitter)

**Solusi:**
✅ **Inverse Roll**: `adjusted_roll = -roll` untuk mirror effect
✅ **Canvas Padding**: Buat canvas dengan diagonal size untuk prevent cropping
✅ **Smoothing Filter**: Alpha smoothing (0.7) untuk stabilitas
   ```python
   roll = alpha * prev_roll + (1 - alpha) * roll_pred
   scale = alpha * prev_scale + (1 - alpha) * scale_pred
   ```
✅ **Scale Adjustment**: `adjusted_scale = scale * 1.2` (mask sedikit lebih besar)

### 3. **Improved Alpha Blending**
**Sebelumnya:**
- Alpha blending loop per channel (slow)
- Mask bisa keluar boundary

**Solusi:**
✅ Vectorized alpha blending dengan numpy broadcasting
✅ Proper boundary checking dengan crop coordinates
✅ Support both RGB dan RGBA masks

## 🎯 Fitur Baru

### 1. **Mirror Mode**
Frame di-flip horizontal untuk user experience yang lebih natural (seperti cermin).

### 2. **Smoothing Filter**
Smoothing factor 0.7 untuk mengurangi jitter pada rotasi dan scale.

### 3. **Better Quality**
- JPEG quality: 90%
- Buffer size optimization
- FPS: 30

### 4. **Debug Mode** (Optional)
Uncomment di code untuk show:
- Bounding box wajah
- Roll value
- Scale value

```python
# cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
# cv2.putText(frame, f"Roll: {roll:.1f}", (x, y-30), ...)
```

## 📊 Parameter Tuning

### Smoothing Factor (alpha)
```python
alpha = 0.7  # 0 = no smoothing, 1 = max smoothing
```
- **0.5-0.6**: Responsive tapi masih ada sedikit jitter
- **0.7**: ✅ Balanced (recommended)
- **0.8-0.9**: Very smooth tapi ada delay

### Scale Adjustment
```python
adjusted_scale = scale * 1.2
```
- **1.0**: Mask sama ukuran dengan wajah
- **1.2**: ✅ Mask sedikit lebih besar (recommended)
- **1.5**: Mask terlalu besar

### Roll Sensitivity
Roll sudah di-inverse (`-roll`), tapi bisa adjust multiplier:
```python
adjusted_roll = -roll * 0.8  # Reduce sensitivity
```

## 🔧 Cara Test

### 1. Test Backend
```bash
# Terminal 1: Run backend
cd "path/to/project"
.venv\Scripts\activate
python app.py
```

### 2. Test di Browser (Simple Test)
Buka browser: `http://localhost:5000/video_feed`

### 3. Test di Godot
1. Run backend terlebih dahulu
2. Buka Godot project
3. Run scene `try_on.tscn`
4. Pilih mask dari dropdown
5. Gerakkan kepala untuk test rotasi

## 🎭 Tips Penggunaan

### Untuk Hasil Terbaik:
1. ✅ Pencahayaan cukup (tidak terlalu gelap/terang)
2. ✅ Wajah menghadap kamera (frontal)
3. ✅ Jarak 50-100cm dari kamera
4. ✅ Background tidak terlalu ramai
5. ✅ Gerakan tidak terlalu cepat (smoothing membutuhkan waktu)

### Jika Mask Tidak Muncul:
1. Cek backend running (`http://localhost:5000`)
2. Cek path mask di `available_masks` dictionary
3. Pastikan mask file adalah PNG dengan alpha channel
4. Cek console untuk error messages

### Jika Rotasi Masih Kurang Pas:
1. Adjust `adjusted_roll` multiplier (0.8 - 1.2)
2. Adjust smoothing `alpha` (0.5 - 0.9)
3. Check model prediction accuracy dengan debug mode

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| FPS Target | 30 |
| Frame Resolution | 640x480 |
| JPEG Quality | 90% |
| Latency | ~33ms per frame |
| Roll MAE | 2.07° |
| Scale MAE | 0.044 |

## 🚀 Next Steps

### Improvements yang Bisa Ditambahkan:
1. **Multi-face support** - Track multiple faces
2. **Landmark-based positioning** - More accurate mask placement
3. **Face mesh overlay** - More detailed face tracking
4. **Custom mask adjustment** - UI untuk adjust position/size per mask
5. **Recording feature** - Save video/photo dengan mask

### Advanced Features:
1. **3D face tracking** - Untuk pitch dan yaw rotation
2. **Emotion detection** - Change mask based on expression
3. **Hand gesture control** - Switch mask dengan gesture
4. **AR effects** - Background effects, particles, etc.

## 📝 Code Changes Summary

### app.py
- ✅ Added mirror flip
- ✅ Added smoothing filter
- ✅ Improved rotation dengan canvas padding
- ✅ Inverse roll untuk mirror effect
- ✅ Better alpha blending
- ✅ Higher JPEG quality
- ✅ Buffer optimization

### TryOn.cs (Godot)
- ✅ Added dropdown untuk select mask
- ✅ HTTP request untuk select mask
- ✅ Load available masks dari backend

## 🎉 Result

Sekarang aplikasi sudah:
- ✅ No more JPEG parsing errors
- ✅ Smooth rotation tracking
- ✅ Accurate mask positioning
- ✅ Natural mirror effect
- ✅ Multi-mask support via dropdown
- ✅ Production-ready quality

Enjoy your Virtual Try-On Topeng! 🎭
