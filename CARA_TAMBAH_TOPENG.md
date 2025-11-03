# 🎭 Cara Menambahkan Topeng/Mask Baru

## 📝 Tutorial Lengkap

### Langkah 1: Siapkan File Gambar Topeng

1. **Format File**: PNG dengan alpha channel (transparansi)
2. **Resolusi**: Minimal 512x512 pixel untuk kualitas baik
3. **Background**: Harus transparan (alpha channel)
4. **Nama File**: Gunakan nama yang deskriptif, contoh:
   - `mask_batman.png`
   - `mask_ironman.png`
   - `topeng_malang_1.png`
   - `topeng_reog.png`

### Langkah 2: Simpan File ke Folder Assets

Letakkan file PNG di folder:
```
godot_project_virtual-try-on/assets/filter_image/
```

Contoh struktur:
```
godot_project_virtual-try-on/
└── assets/
    └── filter_image/
        ├── topeng_1.png           (sudah ada)
        ├── topeng_2.png           (sudah ada)
        ├── badut.png              (sudah ada)
        ├── spiderman.png          (sudah ada)
        ├── ultraman.png           (sudah ada)
        ├── kamen_rider.png        (sudah ada)
        ├── mask_batman.png        (baru)
        ├── mask_spiderman.png     (baru)
        └── topeng_malang_red.png  (baru)
```

### Langkah 3: Daftarkan Mask di Backend (app.py)

Buka file `app.py` di root project, cari baris ini (sekitar line 26-30):

```python
available_masks = {
    "no_filter": None,  # Tidak ada topeng
    "topeng_1": "godot_project_virtual-try-on/assets/filter_image/topeng_1.png",
    "topeng_2": "godot_project_virtual-try-on/assets/filter_image/topeng_2.png",
    "badut": "godot_project_virtual-try-on/assets/filter_image/badut.png",
    "spiderman": "godot_project_virtual-try-on/assets/filter_image/spiderman.png",
    "ultraman": "godot_project_virtual-try-on/assets/filter_image/ultraman.png",
    "kamen_rider": "godot_project_virtual-try-on/assets/filter_image/kamen_rider.png",
}
```

Tambahkan mask baru dengan format:
```python
available_masks = {
    "no_filter": None,  # Tidak ada topeng
    "topeng_1": "godot_project_virtual-try-on/assets/filter_image/topeng_1.png",
    "topeng_2": "godot_project_virtual-try-on/assets/filter_image/topeng_2.png",
    "badut": "godot_project_virtual-try-on/assets/filter_image/badut.png",
    "spiderman": "godot_project_virtual-try-on/assets/filter_image/spiderman.png",
    "ultraman": "godot_project_virtual-try-on/assets/filter_image/ultraman.png",
    "kamen_rider": "godot_project_virtual-try-on/assets/filter_image/kamen_rider.png",
    "batman": "godot_project_virtual-try-on/assets/filter_image/mask_batman.png",
    "spiderman": "godot_project_virtual-try-on/assets/filter_image/mask_spiderman.png",
    "topeng_malang_red": "godot_project_virtual-try-on/assets/filter_image/topeng_malang_red.png",
}
```

**Format:**
- **Key** (sebelum `:`) = Nama yang tampil di dropdown Godot
- **Value** (setelah `:`) = Path lengkap ke file PNG

### Langkah 4: Restart Backend

Setelah menambahkan mask di `app.py`:

1. **Stop backend** yang sedang running (CTRL+C di terminal)
2. **Start ulang** dengan cara:
   
   **Windows PowerShell:**
   ```powershell
   cd "d:\KULIAT\SEMESTER 5\PengolahanCitraDigital\virtual_try-on_topeng_godot"
   .\.venv\Scripts\Activate.ps1
   python app.py
   ```
   
   **Atau double-click:**
   ```
   start_backend.bat
   ```

3. Tunggu sampai muncul: `Uvicorn running on http://0.0.0.0:5000`

### Langkah 5: Test di Godot

1. **Jalankan Godot project**
2. **Run scene** `try_on.tscn`
3. **Klik dropdown** "MaskDropdown" di tengah layar
4. **Pilih mask baru** yang sudah ditambahkan
5. Mask akan langsung aktif dan mengikuti wajah

## 🎨 Tips Membuat Topeng Yang Bagus

### 1. Transparansi
```
✅ BENAR: Background transparan (alpha channel)
❌ SALAH: Background putih/hitam solid
```

### 2. Ukuran & Proporsi
- Topeng wajah penuh: 512x512 atau 1024x1024
- Aksesoris kecil (hidung, kumis): 256x256 sudah cukup
- Aspect ratio: sebaiknya mendekati 1:1 (square)

### 3. Area Aktif
- Fokuskan detail di area tengah (wajah)
- Beri padding di tepi untuk rotasi smooth
- Jangan terlalu detail di edge karena bisa terpotong

### 4. Testing
- Test dengan berbagai pose (miring kanan/kiri)
- Test dengan jarak kamera berbeda
- Adjust parameter `scale` di backend jika perlu

## 🔧 Troubleshooting

### Mask tidak muncul di dropdown
**Solusi:**
1. Cek nama key di `available_masks` tidak ada typo
2. Restart backend setelah edit `app.py`
3. Cek console Godot untuk error messages

### Mask tidak pas dengan wajah
**Solusi:**
1. **Terlalu kecil/besar**: Edit parameter `adjusted_scale` di function `apply_mask_to_face()` (app.py line 60-61)
   ```python
   adjusted_scale = max(0.8, min(1.5, scale * 1.2))  # Adjust multiplier
   ```

2. **Rotasi tidak pas**: Edit parameter `adjusted_roll` (app.py line 75)
   ```python
   adjusted_roll = -roll  # Ubah multiplier jika perlu
   ```

### Mask terlihat patah-patah
**Solusi:**
- Increase JPEG quality di endpoint `/get_frame` (app.py line 276)
  ```python
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 90-100
  ```

### Performance lambat
**Solusi:**
1. Kurangi resolusi mask (resize ke 512x512)
2. Reduce frame rate di TryOn.cs (line 17)
   ```csharp
   private const double FrameInterval = 0.05; // Slower = 20 FPS
   ```

## 📋 Checklist Penambahan Mask Baru

- [ ] File PNG dengan transparansi siap
- [ ] File disimpan di `godot_project_virtual-try-on/assets/filter_image/`
- [ ] Mask didaftarkan di `available_masks` di `app.py`
- [ ] Backend di-restart
- [ ] Test di Godot project
- [ ] Mask muncul di dropdown
- [ ] Mask mengikuti rotasi kepala dengan baik
- [ ] Mask tidak keluar dari batas wajah

## 🎯 Contoh Lengkap

Misalkan ingin menambahkan 3 topeng Malang:

**1. Siapkan files:**
```
topeng_malang_merah.png
topeng_malang_putih.png
topeng_malang_hitam.png
```

**2. Copy ke folder:**
```
godot_project_virtual-try-on/assets/filter_image/
```

**3. Edit app.py:**
```python
available_masks = {
    "no_filter": None,  # Tidak ada topeng
    "topeng_1": "godot_project_virtual-try-on/assets/filter_image/topeng_1.png",
    "topeng_2": "godot_project_virtual-try-on/assets/filter_image/topeng_2.png",
    "badut": "godot_project_virtual-try-on/assets/filter_image/badut.png",
    "spiderman": "godot_project_virtual-try-on/assets/filter_image/spiderman.png",
    "ultraman": "godot_project_virtual-try-on/assets/filter_image/ultraman.png",
    "kamen_rider": "godot_project_virtual-try-on/assets/filter_image/kamen_rider.png",
    "Topeng Malang Merah": "godot_project_virtual-try-on/assets/filter_image/topeng_malang_merah.png",
    "Topeng Malang Putih": "godot_project_virtual-try-on/assets/filter_image/topeng_malang_putih.png",
    "Topeng Malang Hitam": "godot_project_virtual-try-on/assets/filter_image/topeng_malang_hitam.png",
}
```

**4. Restart backend dan test!**

---

## 🚀 Advanced: Adjust Per-Mask Settings

Jika ingin setting berbeda untuk setiap mask (scale, position, dll), bisa expand structure:

```python
available_masks = {
    "no_filter": None,  # Tidak ada topeng
    "topeng_1": {
        "path": "godot_project_virtual-try-on/assets/filter_image/topeng_1.png",
        "scale": 1.0,
        "offset_x": 0,
        "offset_y": 10
    },
    "badut": {
        "path": "godot_project_virtual-try-on/assets/filter_image/badut.png",
        "scale": 1.2,  # Lebih besar
        "offset_x": 0,
        "offset_y": -5  # Sedikit ke atas
    }
}
```

Kemudian adjust code di `apply_mask_to_face()` untuk support dictionary format.

---

**Selamat mencoba! 🎭**
