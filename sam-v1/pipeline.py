from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import numpy as np
import json
import os 

with open('config.json', 'r') as f:
    config = json.load(f)

input_image_path = config.get('input_image_path')  
output_folder = config.get('output_folder')
img_name = os.path.splitext(os.path.basename(input_image_path))[0]
output_folder = os.path.join(output_folder, img_name)
os.makedirs(output_folder, exist_ok=True)




sam = sam_model_registry["vit_h"](checkpoint="checkpoint/sam_vit_h_4b8939.pth").to('cuda')


mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

# Read img 
image = cv2.imread(input_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)





input_point = np.array([[586, 716]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
 point_coords=input_point,
 point_labels=input_label,
 multimask_output=True,
)




output_mask = masks[0]
output_mask = (output_mask * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_folder, f'{img_name}-output-mask.jpg'), output_mask)

