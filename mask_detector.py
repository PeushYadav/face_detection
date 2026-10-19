"""Shared face detection + mask classification logic used by the image and webcam scripts."""
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
CLASSES = ["Mask", "No Mask"]
COLORS = [(0, 200, 0), (0, 0, 230)]  # BGR: green for mask, red for no mask


class MaskDetector:
    def __init__(self, model_path="models/mask_detector.keras", face_dir="face_detector", confidence=0.5):
        prototxt = os.path.join(face_dir, "deploy.prototxt")
        weights = os.path.join(face_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.face_net = cv2.dnn.readNet(prototxt, weights)  # OpenCV SSD ResNet-10 face detector
        self.mask_net = tf.keras.models.load_model(model_path)
        self.confidence = confidence
        # compiled graph: ~5x faster than model.predict() for the small batches a video frame produces
        self._classify = tf.function(
            lambda x: self.mask_net(x, training=False),
            input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)])

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        boxes = []
        # detections are sorted by confidence, highest first
        for i in range(detections.shape[2]):
            if detections[0, 0, i, 2] < self.confidence:
                continue
            rel = detections[0, 0, i, 3:7]
            # this SSD emits phantom boxes reaching far past the frame on close-up faces
            if rel.min() < -0.1 or rel.max() > 1.1:
                continue
            x1, y1, x2, y2 = (rel * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 - x1 > 10 and y2 - y1 > 10 and not any(self._mostly_inside((x1, y1, x2, y2), b) for b in boxes):
                boxes.append((x1, y1, x2, y2))
        return boxes

    @staticmethod
    def _mostly_inside(a, b, thresh=0.5):
        """True if more than `thresh` of box a's area overlaps box b (duplicate detection of the same face)."""
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            return False
        return iw * ih > thresh * (a[2] - a[0]) * (a[3] - a[1])

    def predict(self, frame):
        """Return a list of (box, label_index, probability) for every face in a BGR frame."""
        boxes = self.detect_faces(frame)
        if not boxes:
            return []
        faces = []
        for x1, y1, x2, y2 in boxes:
            face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype("float32")
            faces.append(preprocess_input(face))
        probs = self._classify(np.array(faces, dtype="float32")).numpy()
        return [(box, int(p.argmax()), float(p.max())) for box, p in zip(boxes, probs)]

    @staticmethod
    def draw(frame, results):
        for (x1, y1, x2, y2), idx, prob in results:
            color = COLORS[idx]
            text = f"{CLASSES[idx]}: {prob * 100:.1f}%"
            scale = min(1.0, max(0.5, (x2 - x1) / 220))  # label size follows face size
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            ty = max(y1, th + 10)
            cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 6, ty), color, -1)
            cv2.putText(frame, text, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        return frame
