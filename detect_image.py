"""Detect face masks in a still image.

Usage:
    python detect_image.py --image path/to/photo.jpg [--output result.jpg] [--no-show]
"""
import argparse

import cv2

from mask_detector import CLASSES, MaskDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="models/mask_detector.keras")
    ap.add_argument("--confidence", type=float, default=0.5, help="min face-detection confidence")
    ap.add_argument("--output", default=None, help="save the annotated image here")
    ap.add_argument("--no-show", action="store_true", help="don't open a preview window")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    detector = MaskDetector(args.model, confidence=args.confidence)
    results = detector.predict(image)
    if not results:
        print("No faces detected.")
    for i, ((x1, y1, x2, y2), idx, prob) in enumerate(results, 1):
        print(f"Face {i} at ({x1},{y1},{x2},{y2}): {CLASSES[idx]} ({prob * 100:.1f}%)")

    detector.draw(image, results)
    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Saved annotated image to {args.output}")
    if not args.no_show:
        cv2.imshow("Face Mask Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
