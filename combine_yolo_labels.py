import os, json, cv2

def yolo_to_bbox(txt_file, img_shape):
    h_img, w_img = img_shape[:2]
    with open(txt_file, "r") as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, width, height = map(float, parts)
        xmin = int((x_center - width / 2) * w_img)
        ymin = int((y_center - height / 2) * h_img)
        w = int(width * w_img)
        h = int(height * h_img)
        bboxes.append([xmin, ymin, w, h])
    return bboxes

def process_split(img_dir, label_dir, output_json):
    data = []
    img_files = os.listdir(img_dir)
    for img_name in img_files:
        if not img_name.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        boxes = yolo_to_bbox(txt_path, img.shape)
        for bbox in boxes:
            data.append({
                "file": img_name,
                "bbox": bbox,
                "roll": 0.0,
                "scale": 1.0
            })
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} annotations to {output_json}")

# Ganti path sesuai dataset kamu
process_split(
    "dataset/train/images",
    "dataset/labels/train",
    "dataset/train/labels.json"
)

process_split(
    "dataset/val/images",
    "dataset/labels/val",
    "dataset/val/labels.json"
)
