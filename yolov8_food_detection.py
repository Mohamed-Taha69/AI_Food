"""
YOLOv8 Food Detection & Cropping — Google Colab Script
=======================================================
Detects food items in Food-101 images using a pre-trained YOLOv8 model,
crops the detected regions, and saves them into a structured directory
that mirrors the original class subfolders.

Usage (run each section as a Colab cell):
  1. Install dependencies
  2. Run the detection & cropping pipeline
"""

# ============================================================
# Cell 1 — Install dependencies
# ============================================================
# !pip install ultralytics tqdm

# ============================================================
# Cell 2 — Imports & configuration
# ============================================================
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────
IMAGES_DIR = Path("/content/food-101/images")  # Food-101 images root
CROPS_DIR  = Path("/content/crops")            # Output directory

# ── Model ────────────────────────────────────────────────────
MODEL_NAME = "yolov8n.pt"  # Nano variant — fast & lightweight
CONFIDENCE = 0.30          # Minimum confidence threshold

# ── COCO food-related class IDs (YOLOv8 / COCO 80-class) ───
# 46: banana, 47: apple, 48: sandwich, 49: orange,
# 50: broccoli, 51: carrot, 52: hot dog, 53: pizza,
# 54: donut, 55: cake
FOOD_CLASS_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}

# ============================================================
# Cell 3 — Detection & cropping pipeline
# ============================================================

def detect_and_crop():
    """Run YOLOv8 detection on every Food-101 image and save crops."""

    # Load the pre-trained YOLOv8 model
    model = YOLO(MODEL_NAME)
    print(f"✅ Loaded model: {MODEL_NAME}")

    # Collect all image paths (Food-101 uses class subfolders)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [
        p for p in IMAGES_DIR.rglob("*")
        if p.suffix.lower() in image_extensions
    ]
    print(f"📂 Found {len(all_images):,} images across "
          f"{len(set(p.parent.name for p in all_images))} classes\n")

    total_crops = 0
    skipped     = 0

    for img_path in tqdm(all_images, desc="Processing images", unit="img"):
        # Read image with OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        # Preserve original subfolder structure  (e.g. pizza/, steak/)
        class_name = img_path.parent.name
        output_dir = CROPS_DIR / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference
        results = model.predict(source=img, conf=CONFIDENCE, verbose=False)

        # Process detections
        crop_idx = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                # Optional: filter to food-related classes only
                # Uncomment the next two lines to restrict to COCO food classes
                # if cls_id not in FOOD_CLASS_IDS:
                #     continue

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to image boundaries
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Skip degenerate boxes
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                # Crop and save
                crop = img[y1:y2, x1:x2]
                conf  = float(box.conf[0])
                label = model.names[cls_id]

                crop_name = f"{img_path.stem}_crop{crop_idx}_{label}_{conf:.2f}.jpg"
                cv2.imwrite(str(output_dir / crop_name), crop)
                crop_idx += 1

        total_crops += crop_idx

    # ── Summary ──────────────────────────────────────────────
    num_classes = len(list(CROPS_DIR.iterdir())) if CROPS_DIR.exists() else 0
    print(f"\n{'='*50}")
    print(f"✅ Done!")
    print(f"   Images processed : {len(all_images) - skipped:,}")
    print(f"   Images skipped   : {skipped:,}")
    print(f"   Total crops saved: {total_crops:,}")
    print(f"   Output classes   : {num_classes}")
    print(f"   Crops directory  : {CROPS_DIR}")
    print(f"{'='*50}")


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    detect_and_crop()
