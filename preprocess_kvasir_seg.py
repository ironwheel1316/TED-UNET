import os
import cv2
import numpy as np
from glob import glob
import argparse


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_kvasir_seg(input_dir, output_dir, target_size=(288, 288)):
    """
    Preprocess Kvasir-SEG dataset by resizing images and masks to target size.
    
    Args:
        input_dir (str): Path to the input directory containing Kvasir-SEG dataset.
        output_dir (str): Path to save the preprocessed dataset.
        target_size (tuple): Target size for resizing images and masks.
    """
    # Define paths
    image_dir = os.path.join(input_dir, 'images')
    mask_dir = os.path.join(input_dir, 'masks')
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_mask_dir = os.path.join(output_dir, 'masks',"0")
    
    # Ensure output directories exist
    ensure_dir(output_image_dir)
    ensure_dir(output_mask_dir)
    
    # Get list of image files
    image_files = glob(os.path.join(image_dir, '*.jpg'))
    image_ids = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
    
    print(f"Found {len(image_ids)} images in {image_dir}")
    
    for img_id in image_ids:
        # Read image
        img_path = os.path.join(image_dir, img_id + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_path = os.path.join(mask_dir, img_id + '.jpg')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Resize image and mask
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Save preprocessed image and mask
        output_img_path = os.path.join(output_image_dir, img_id + '.png')
        output_mask_path = os.path.join(output_mask_dir, img_id + '.png')
        
        cv2.imwrite(output_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_mask_path, mask_resized)
        
        print(f"Processed {img_id}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Kvasir-SEG dataset.')
    parser.add_argument('--input_dir', type=str, default='inputs/Kvasir-SEG', 
                        help='Input directory containing Kvasir-SEG dataset.')
    parser.add_argument('--output_dir', type=str, default='inputs/kvasir_seg_288', 
                        help='Output directory to save preprocessed dataset.')
    parser.add_argument('--size', type=int, default=288, 
                        help='Target size for resizing images and masks.')
    
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    preprocess_kvasir_seg(args.input_dir, args.output_dir, target_size)
    print(f"Preprocessing completed. Data saved to {args.output_dir}")

if __name__ == '__main__':
    main()

