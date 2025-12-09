# Hướng dẫn Retrain Model cho Black Speech Bubbles

## Tổng quan

Model YOLO hiện tại (`model/model.pt`) chủ yếu được train trên bóng thoại trắng. Để cải thiện detection cho bóng thoại đen, bạn cần retrain model với dataset bao gồm cả black bubbles.

## Bước 1: Chuẩn bị Dataset

### 1.1 Thu thập ảnh

Cần thu thập các trang manga/manhwa có chứa:
- Bóng thoại trắng (đã có trong dataset hiện tại)
- **Bóng thoại đen** với chữ trắng
- Bóng thoại có viền (outline)

**Nguồn ảnh:**
- Manga có dark theme (horror, action)
- Manhwa với các scene tối
- One Piece, Attack on Titan, Tokyo Ghoul (có nhiều black bubbles)

### 1.2 Annotate dữ liệu

Sử dụng tool annotation như:
- **Roboflow** (recommend - dễ dùng, export YOLO format)
- **LabelImg** (free, offline)
- **CVAT** (online, team collaboration)

**Label:**
- Class: `speech_bubble` (giống dataset hiện tại)

## Bước 2: Cấu trúc Dataset

```
dataset/
├── train/
│   ├── images/
│   │   ├── page001.jpg
│   │   └── ...
│   └── labels/
│       ├── page001.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

**data.yaml:**
```yaml
train: ./train/images
val: ./valid/images
nc: 1
names: ['speech_bubble']
```

## Bước 3: Training

Mở notebook `model/model_training.ipynb` hoặc chạy:

```python
from ultralytics import YOLO

# Load model hiện tại (transfer learning)
model = YOLO('model/model.pt')

# Train với dataset mới
results = model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    save=True,
    project='runs/train',
    name='black_bubbles'
)
```

**Lưu ý:**
- `epochs=50`: Có thể tăng nếu model chưa converge
- `patience=10`: Early stopping
- Transfer learning từ model cũ giúp giữ khả năng detect white bubbles

## Bước 4: Evaluate và Replace Model

```python
# Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")

# Export model mới
model.export(format='pt')
```

Copy model mới vào `model/model.pt`:
```bash
cp runs/train/black_bubbles/weights/best.pt model/model.pt
```

## Tips

1. **Tỷ lệ dataset:**
   - 70% white bubbles + 30% black bubbles
   - Tránh imbalance quá lớn

2. **Augmentation:**
   - YOLO tự động augment
   - Có thể thêm invert colors để tăng variety

3. **Số lượng tối thiểu:**
   - 100+ ảnh với black bubbles
   - 500+ annotations

## Fallback hiện tại

Trong khi chờ retrain, hệ thống đã có **OpenCV fallback** để detect black bubbles:
- File: `detect_bubbles.py` → `detect_black_bubbles()`
- Detect dựa vào contour và intensity
- Tự động merge với YOLO results
