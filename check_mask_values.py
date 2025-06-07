import os
import cv2
import numpy as np
from glob import glob

def check_mask_values(mask_dir, mask_ext='.png', num_samples=10):

    
    mask_paths = glob(os.path.join(mask_dir, '*' + mask_ext))
    if not mask_paths:
        print(f"No masks found in {mask_dir}")
        return
    
    print(f"Found {len(mask_paths)} masks in {mask_dir}")
    for i, mask_path in enumerate(mask_paths[:num_samples]):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            continue
        
        unique_values = np.unique(mask)
        print(f"Mask {i+1}: {os.path.basename(mask_path)}")
        print(f"Unique values in mask: {unique_values}")
        print(f"Number of unique values: {len(unique_values)}")
        print(f"Shape of mask: {mask.shape}")
        print("---")

if __name__ == "__main__":
    mask_dir = "inputs/cityscape_96/masks"
    check_mask_values(mask_dir)
