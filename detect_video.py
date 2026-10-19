"""Real-time face mask detection from a webcam (or a video file).

Usage:
    python detect_video.py              # default webcam
    python detect_video.py --source 1   # another camera
    python detect_video.py --source clip.mp4
Press 'q' to quit.
"""
import argparse
import time

import cv2

from mask_detector import MaskDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--model", default="models/mask_detector.keras")
    ap.add_argument("--confidence", type=float, default=0.5, help="min face-detection confidence")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    detector = MaskDetector(args.model, confidence=args.confidence)
    print("[INFO] running... press 'q' to quit")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1) if isinstance(source, int) else frame
        detector.draw(frame, detector.predict(frame))

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Face Mask Detection - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
