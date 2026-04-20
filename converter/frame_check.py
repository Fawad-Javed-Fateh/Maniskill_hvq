import os
import cv2

video_path = "/home/retrocausal-train/Desktop/Maniskill_hvq/converter/robot/good cycles/episode_0008/20260408_151633.mp4"

# Crop coordinates
x1, y1 = 337, 54
x2, y2 = 575, 473

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(x2 - x1)
height = int(y2 - y1)

# Output path in the same directory
out_path = os.path.join(os.path.dirname(video_path), "cropped.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cropped = frame[y1:y2, x1:x2]
    out.write(cropped)

cap.release()
out.release()

print(f"Cropped video saved at: {out_path}")