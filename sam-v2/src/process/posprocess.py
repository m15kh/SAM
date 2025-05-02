
import cv2
import os
import numpy as np
import subprocess
from tqdm import tqdm

def generate_segmented_video(input_video_path, output_video_path, video_segments, alpha=0.5):
    """
    Generate a segmented video by overlaying masks on the original video frames.
    
    Parameters:
    -----------
    input_video_path : str
        Path to the input video file
    output_video_path : str
        Path where the output video will be saved
    video_segments : dict
        Dictionary mapping frame indices to segmentation masks
        Format: {frame_idx: {obj_id: mask}}
    alpha : float, optional
        Transparency factor for the mask overlay (default: 0.5)
    
    Returns:
    --------
    str
        Path to the output segmented video
    """
    # Import tqdm for progress bar
    
    # Function to overlay mask on the frame
    def apply_mask_to_frame(frame, mask, alpha=0.5):
        """
        Overlay the mask on the original frame. The mask is a binary mask (0/1),
        and the frame is an image (H x W x 3).
        
        frame: numpy array (H x W x 3)
        mask: binary numpy array (H x W)
        alpha: transparency factor for the mask
        """
        # Ensure the mask is (H, W, 1) shape, and expand it to (H, W, 3)
        mask_rgb = np.expand_dims(mask, axis=-1)  # Convert (H, W) to (H, W, 1)
        mask_rgb = np.repeat(mask_rgb, 3, axis=-1)  # Expand to (H, W, 3)
        
        # Color the mask (Green color as default)
        mask_rgb = mask_rgb * np.array([0, 255, 0])  # Green mask (you can change the color)

        # Ensure the frame is in the correct shape (H, W, 3)
        frame_rgb = frame.astype(np.uint8)

        # Check that both frame and mask_rgb have the same size and number of channels
        if frame_rgb.shape != mask_rgb.shape:
            raise ValueError(f"Frame and mask size do not match: {frame_rgb.shape} != {mask_rgb.shape}")

        # Overlay the mask on the original frame with transparency
        return cv2.addWeighted(frame_rgb, 1 - alpha, mask_rgb.astype(np.uint8), alpha, 0)

    # Get video properties
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change this based on your needs
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Iterate over the video frames with tqdm progress bar
    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply the mask for the current frame
            if frame_idx in video_segments:
                # Get the mask for the current frame
                frame_mask = np.zeros((frame_height, frame_width), dtype=np.float32)
                for obj_id, mask in video_segments[frame_idx].items():
                    # Check for extra dimensions and remove them
                    if mask.ndim > 2:
                        mask = np.squeeze(mask)  # Remove singleton dimensions
                    
                    # Ensure mask matches frame dimensions
                    if mask.shape[0] == frame_height and mask.shape[1] == frame_width:
                        # Overlay all object masks on top of each other
                        frame_mask = np.maximum(frame_mask, mask)
                    else:
                        print(f"Skipping mask with incompatible shape: {mask.shape}")
                
                # Apply the mask to the frame (overlay it with transparency)
                frame_with_mask = apply_mask_to_frame(frame, frame_mask, alpha)
                
                # Write the frame with the mask to the output video
                out.write(frame_with_mask)
            else:
                # Write the original frame if no mask is available
                out.write(frame)
            
            # Update progress bar
            pbar.update(1)
            frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    
    return output_video_path


if __name__ == "__main__":
    # Example usage:
    input_video_path = video_file
    output_video_path = "segmented_output_video.mp4"

    # Call the function with your video segments
    segmented_video_path = generate_segmented_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        video_segments=video_segments
    )

    print(f"Segmented video saved as {segmented_video_path}")