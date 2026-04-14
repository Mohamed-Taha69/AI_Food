# 🍕 YOLOv8 Food Detection & Cropping — Food-101

Detect food items in [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) images using a pre-trained **YOLOv8** model, crop them, and prepare structured data for a downstream classification model.

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.  
Make sure the runtime is set to **GPU** for faster inference:  
`Runtime → Change runtime type → T4 GPU`

---

### Step 2 — Install dependencies

```python
!pip install ultralytics tqdm
```

---

### Step 3 — Download & extract the Food-101 dataset

```python
!wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
!tar -xzf food-101.tar.gz -C /content/
```

This creates `/content/food-101/images/` with 101 class subfolders.

---

### Step 4 — Upload the script

**Option A** — Upload manually:  
Use the Colab file browser (📁 icon on the left) to upload `yolov8_food_detection.py` to `/content/`.

**Option B** — Clone from your repo (if applicable):
```python
!git clone https://github.com/Mohamed-Taha69/AI_Food.git
%cd AI_Food
```

---

### Step 5 — Run the detection & cropping pipeline

```python
%run yolov8_food_detection.py
```

Or import and call directly:

```python
from yolov8_food_detection import detect_and_crop
detect_and_crop()
```

> ⏱️ **Time estimate:** ~2–3 hours for the full 101k images on a T4 GPU.  
> 💡 **Tip:** Test on a small subset first (see section below).

---

### Step 6 — Verify the output

```python
import os

crops_dir = "/content/crops"
classes = sorted(os.listdir(crops_dir))
print(f"Classes: {len(classes)}")

for cls in classes[:5]:
    n = len(os.listdir(os.path.join(crops_dir, cls)))
    print(f"  {cls}: {n} crops")
```

---

### Step 7 — Download the crops (optional)

```python
!zip -r /content/food_101_crops.zip /content/crops
```

Then download via the Colab file browser or:

```python
from google.colab import files
files.download("/content/food_101_crops.zip")
```

---

## 🧪 Test on a Small Subset First

To do a quick test before processing all 101k images:

```python
import shutil, os, random

src = "/content/food-101/images"
dst = "/content/food-101-mini/images"

# Copy 10 images from 3 random classes
for cls in random.sample(os.listdir(src), 3):
    cls_src = os.path.join(src, cls)
    cls_dst = os.path.join(dst, cls)
    os.makedirs(cls_dst, exist_ok=True)
    imgs = random.sample(os.listdir(cls_src), min(10, len(os.listdir(cls_src))))
    for img in imgs:
        shutil.copy2(os.path.join(cls_src, img), cls_dst)

print("Mini dataset ready!")
```

Then edit the paths in the script:

```python
IMAGES_DIR = Path("/content/food-101-mini/images")
CROPS_DIR  = Path("/content/crops-mini")
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `yolov8n.pt` | YOLOv8 variant (`n`/`s`/`m`/`l`/`x`) |
| `CONFIDENCE` | `0.30` | Minimum detection confidence |
| `IMAGES_DIR` | `/content/food-101/images` | Input images root |
| `CROPS_DIR` | `/content/crops` | Output crops root |

### Model variants (speed vs accuracy)

| Model | Size | Speed | mAP |
|-------|------|-------|-----|
| `yolov8n.pt` | 6 MB | ⚡ Fastest | Good |
| `yolov8s.pt` | 22 MB | Fast | Better |
| `yolov8m.pt` | 52 MB | Medium | Great |
| `yolov8l.pt` | 87 MB | Slow | Excellent |

---

## 📁 Output Structure

```
/content/crops/
├── pizza/
│   ├── 1008104_crop0_pizza_0.85.jpg
│   └── 1008104_crop1_bowl_0.42.jpg
├── steak/
│   ├── 100274_crop0_dining table_0.67.jpg
│   └── ...
└── sushi/
    └── ...
```

Each crop filename encodes: `{original_name}_crop{N}_{detected_label}_{confidence}.jpg`

---

## 🔍 Food-Only Filtering

By default, the script detects **all** 80 COCO classes (plates, bowls, knives, etc. can help isolate the food region). To detect only COCO food categories (banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake), uncomment lines **83–84** in the script:

```python
if cls_id not in FOOD_CLASS_IDS:
    continue
```

---

## 📋 Next Steps

After cropping, you can use the structured `/content/crops/` directory to:

1. **Train a classification model** (e.g., EfficientNet, ResNet) on the cropped regions
2. **Fine-tune** on a subset of high-confidence crops
3. **Data augmentation** using the crops as a cleaner dataset

```python
# Example: Load crops with PyTorch ImageFolder
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder("/content/crops", transform=transform)
print(f"Classes: {dataset.classes}")
print(f"Total images: {len(dataset)}")
```
