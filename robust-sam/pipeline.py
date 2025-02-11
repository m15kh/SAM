import torch
torch.cuda.empty_cache()

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import torch
import torch.nn as nn

from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide

import json
config_path = "/home/ubuntu/m15kh/3d-point/point_cloud_completion/point_cloud_completion/src/config.json" 

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)    

# Extract paths
data = config["dev-robust-sam"][0] 
image_camera_path = data["image_camera_path"]
checkpoint = data["checkpoint"]
output_folder = data["output_path"] 

def show_boxes(coords, ax):
    x1, y1, x2, y2 = coords
    width = x2-x1
    height = y2-y1
    bbox = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)

def show_points(coords, labels, ax, marker_size=500):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])

    h, w = mask.shape[-2:]
    mask = mask.detach().cpu().numpy()
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
checkpoint_path = '/home/ubuntu/m15kh/3d-point/point_cloud_completion/point_cloud_completion/src/checkpoints/robustsam_checkpoint_l.pth'
model_type = "vit_l"
device = "cuda"
model = sam_model_registry[model_type](opt=None, checkpoint=checkpoint_path)
model.to(device=device)
print('Succesfully loading model from {}'.format(checkpoint_path))

sam_transform = ResizeLongestSide(model.image_encoder.img_size)

image = cv2.imread(image_camera_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(image)

image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(device)
image_t = torch.permute(image_t, (0, 3, 1, 2))
image_t_transformed = sam_transform.apply_image_torch(image_t.float())

prompt = np.array([315, 12, 3240, 5162])
box_t = torch.Tensor(prompt).unsqueeze(0).to(device)

data_dict = {}
data_dict['image'] = image_t_transformed
data_dict['boxes'] = sam_transform.apply_boxes_torch(box_t, image_t.shape[-2:]).unsqueeze(0)
data_dict['original_size'] = image_t.shape[-2:]

with torch.no_grad():
    batched_output = model.predict(None, [data_dict], multimask_output=False, return_logits=False)

output_mask = batched_output[0]['masks']
plt.figure(figsize=(10,10))
plt.imshow(image)
show_boxes(prompt, plt.gca())
show_mask(output_mask[0][0], plt.gca())
plt.axis('off')

# Save the figure
output_image_path = os.path.join(output_folder, "output_image.png")
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
print(f"Image saved to {output_image_path}")
