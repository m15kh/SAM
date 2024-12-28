import cv2
import os
import json
import numpy as np
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch

print("Running Fast-SAM pipeline")

# Step 1: Load configuration file
print("Loading configuration file")
with open('config.json', 'r') as f:
    config = json.load(f)

input_image_path = config.get('input_image_path')
output_folder = config.get('output_folder')
img_name = os.path.splitext(os.path.basename(input_image_path))[0]
output_folder = os.path.join(output_folder, img_name)
checkpoint = config.get('checkpoint')

# Step 2: Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 3: Initialize FastSAM model
print("Initializing model")
model = FastSAM()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Step 4: Perform inference
print("Running inference")
everything_result = model(
    input_image_path,
    device=device,
    imgsz=1024,
    conf=0.4,
    iou=0.9,
    retina_masks=True  # Corrected argument
)

# Step 5: Generate segmentation using point prompt
print("Generating segmentation")
point = tuple(config.get('point', [0, 0]))  # Default to (0, 0) if not provided
point_x, point_y = point
prompt = FastSAMPrompt(input_image_path, everything_result, device=device)
segmentation_mask = prompt.point_prompt([point_x], [point_y])


# segmentation_output_path = os.path.join(output_folder, f"{img_name}_segmentation.png")
# cv2.imwrite(segmentation_output_path, segmentation_mask)

# print(f"Segmentation result saved to {segmentation_output_path}")
