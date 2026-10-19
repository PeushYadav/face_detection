"""Generate the figures used in the presentation and docs (run from the project root).

    python docs/make_assets.py
"""
import json
import os
import random
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from train import CLASSES, load_dataset  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

OUT = "docs/assets"
os.makedirs(OUT, exist_ok=True)
random.seed(7)


def square(img, size):
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def dataset_samples():
    for cls in CLASSES:
        files = [f for f in os.listdir(f"dataset/{cls}") if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        picks = []
        for f in random.sample(files, 60):
            img = cv2.imread(f"dataset/{cls}/{f}")
            if img is not None and min(img.shape[:2]) >= 120:
                picks.append(square(img, 200))
            if len(picks) == 4:
                break
        cv2.imwrite(f"{OUT}/samples_{cls}.png", np.hstack(picks))


def augmentation_demo():
    img = None
    for f in sorted(os.listdir("dataset/with_mask")):
        img = cv2.imread(f"dataset/with_mask/{f}")
        if img is not None and min(img.shape[:2]) >= 250:
            break
    img = square(img, 224)
    h, w = img.shape[:2]
    flip = cv2.flip(img, 1)
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0), (w, h), borderMode=cv2.BORDER_REFLECT)
    zoom = cv2.resize(img[25:199, 25:199], (w, h))
    shift = cv2.warpAffine(img, np.float32([[1, 0, 20], [0, 1, -16]]), (w, h), borderMode=cv2.BORDER_REFLECT)
    for name, im in [("original", img), ("flip", flip), ("rotate", rot), ("zoom", zoom), ("shift", shift)]:
        cv2.imwrite(f"{OUT}/aug_{name}.png", im)


def demo_outputs():
    """Annotate held-out (test split) images so the demo never shows training data."""
    from mask_detector import MaskDetector
    paths, labels = load_dataset("dataset")
    _, x_test, _, y_test = train_test_split(paths, labels, test_size=0.20, stratify=labels, random_state=42)
    det = MaskDetector()
    found = {0: [], 1: []}
    for p, y in zip(x_test, y_test):
        img = cv2.imread(p)
        if img is None or min(img.shape[:2]) < 300 or len(found[y]) >= 3:
            continue
        res = det.predict(img)
        if res and all(idx == y for _, idx, _ in res):
            scale = 640 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale)
            res = [((int(a * scale), int(b * scale), int(c * scale), int(d * scale)), i, pr)
                   for (a, b, c, d), i, pr in res]
            det.draw(img, res)
            found[y].append(img)
        if all(len(v) >= 3 for v in found.values()):
            break
    demo = []
    for y, imgs in found.items():
        for k, im in enumerate(imgs):
            name = f"demo_{CLASSES[y]}_{k}.jpg"
            cv2.imwrite(f"{OUT}/{name}", im)
            demo.append({"file": name, "w": im.shape[1], "h": im.shape[0], "label": CLASSES[y]})
    print("demo images:", {CLASSES[k]: len(v) for k, v in found.items()})

    # measure end-to-end latency (detect + classify) on a 640x480 frame
    import time
    frame = cv2.resize(cv2.imread(x_test[0]), (640, 480))
    det.predict(frame)
    t = time.perf_counter()
    for _ in range(20):
        det.predict(frame)
    ms = (time.perf_counter() - t) / 20 * 1000
    print(f"latency: {ms:.1f} ms/frame (~{1000 / ms:.1f} FPS)")
    return demo, ms


def parse_log(demo, latency_ms, path="models/train_log.txt"):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, encoding="utf-8", errors="ignore").read())
    hist = [dict(zip(["acc", "loss", "val_acc", "val_loss"], map(float, m))) for m in re.findall(
        r"accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)", text)]
    report = {}
    for cls in CLASSES:
        m = re.search(rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", text)
        report[cls] = dict(zip(["precision", "recall", "f1", "support"], map(float, m.groups())))
    report["accuracy"] = float(re.search(r"accuracy\s+([\d.]+)\s+\d+", text).group(1))
    cm = re.search(r"\[\[\s*(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\]\]", text)
    report["confusion"] = [[int(cm.group(1)), int(cm.group(2))], [int(cm.group(3)), int(cm.group(4))]]
    params = {k: int(v.replace(",", "")) for k, v in re.findall(r"(Total|Trainable|Non-trainable) params: ([\d,]+)", text)}
    json.dump({"history": hist, "report": report, "params": params, "demo": demo,
               "latency_ms": round(latency_ms, 1)}, open(f"{OUT}/metrics.json", "w"), indent=2)
    print(json.dumps({"epochs": len(hist), "report": report, "params": params}, indent=1))


if __name__ == "__main__":
    dataset_samples()
    augmentation_demo()
    if os.path.exists("models/mask_detector.keras"):
        parse_log(*demo_outputs())
