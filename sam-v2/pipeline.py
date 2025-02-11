# Load configuration file
import json
import cv2
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

config_path = "/home/ubuntu/m15kh/3d-point/point_cloud_completion/point_cloud_completion/src/config.json" 

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)    

# Extract paths
data = config["dev-sam2"][0] 
image_camera_path = data["image_camera_path"]
checkpoint = data["checkpoint"]
output_folder = data["output_path"] 
model_cfg = data["model_cfg"]

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_path="output_image.png"):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"{save_path}_{i+1}.png")
        plt.close()

# Load image
image = cv2.imread(image_camera_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM2 model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(model_cfg, checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

# Define bounding box coordinates
box_coords = np.array([315, 12, 3240, 5162])

masks, scores, logits = predictor.predict(
    box=box_coords,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

show_masks(image, masks, scores, box_coords=box_coords, borders=True)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_folder, "segmented_image.png")

# Save the result as a PNG file
cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for saving

print(f"Image saved at: {output_file}")