import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


import os
import cv2

import os
import cv2

def extract_video_frames(video_path, output_dir="frames"):
    """Extract frames from a video file and save them as zero-padded JPG images like '00000.jpg'."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {width}x{height} resolution")

    # Extract and save frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")  # Save as '00000.jpg'
        cv2.imwrite(frame_path, frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Extracted {frame_idx}/{frame_count} frames...")

    cap.release()
    print(f"Extracted {frame_idx} frames to '{output_dir}'")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_idx
    }

