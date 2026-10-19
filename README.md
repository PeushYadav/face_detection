# Face Mask Detection — ML Mini Project

Detects whether each person in an image or live webcam feed is wearing a face mask.

## How it works

Two-stage pipeline:

1. **Face detection** — OpenCV's pre-trained SSD (ResNet-10 backbone, `face_detector/`) finds face bounding boxes.
2. **Mask classification** — each face crop (224×224) goes through a **MobileNetV2** CNN pretrained on ImageNet
   (transfer learning) with a new head: `AvgPool → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(2, softmax)`.
   The base is frozen; only the head is trained. Data augmentation (flip, rotate, zoom, shift) reduces overfitting.

Output: green box = **Mask**, red box = **No Mask**, with confidence %.

## Dataset

`dataset/` — 4,092 images in two classes (`with_mask` 2,162, `without_mask` 1,930), from the
open-source [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) repo (MIT).
80/20 stratified train/test split.

## Results

99.2% accuracy on 819 held-out test images (7 errors), about 81 ms per webcam frame on a laptop CPU.
See **PROJECT_EXPLANATION.md** for the full write-up and **Face_Mask_Detection_Presentation.pptx** for the slides.

## Project structure

```
dataset/            with_mask/ and without_mask/ training images
face_detector/      OpenCV SSD face detector (prototxt + caffemodel)
models/             trained model + training plot (created by train.py)
train.py            trains and evaluates the classifier
mask_detector.py    shared face-detection + classification code
detect_image.py     run on a photo
detect_video.py     run on webcam / video in real time
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```powershell
# 1. Train (optional, models/mask_detector.keras is already trained; ~25 min on CPU); saves models/mask_detector.keras and models/training_plot.png
python train.py --epochs 20

# 2. Test on an image
python detect_image.py --image some_photo.jpg --output result.jpg

# 3. Live webcam (press q to quit)
python detect_video.py
```

## Possible extensions

- Add a third class: *mask worn incorrectly* (below nose).
- Fine-tune the top MobileNetV2 layers for higher accuracy.
- Export to TFLite / ONNX for mobile or Raspberry Pi deployment.
- Web UI with Streamlit or Flask.
