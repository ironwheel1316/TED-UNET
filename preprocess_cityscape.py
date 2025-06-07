import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import argparse

# Cityscapes官方调色板 (RGB)，通常对应0-18为训练类别，19为背景/忽略
# 你提供的调色板有20个颜色，将映射到ID 0-19
CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)  # 通常 (0,0,0) 是 unlabelled
]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(img_path, output_dir, img_size=(224, 224)):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Warning: Could not read image {img_path}")
        return None
    img_bgr_resized = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_LINEAR)
    img_rgb_resized = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    
    output_filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb_resized, cv2.COLOR_RGB2BGR))
    return img_rgb_resized

def preprocess_mask(mask_path, output_base_mask_dir, img_size=(224, 224), palette=CITYSCAPES_PALETTE, num_classes=20):
    """
    Preprocesses a single mask.
    Converts a colored mask to multiple binary masks, one for each class ID.
    Saves each binary mask into a subdirectory named after its class ID.

    Args:
        mask_path (str): Path to the original colored mask file.
        output_base_mask_dir (str): Base directory where class-specific mask subdirectories will be created.
                                     e.g., output_base_mask_dir/0/, output_base_mask_dir/1/, ...
        img_size (tuple): Target size for resizing the mask (height, width).
        palette (list): List of RGB tuples representing the color palette.
        num_classes (int): Total number of classes (corresponds to len(palette)).
    """
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        print(f"Warning: Could not read mask {mask_path}")
        return False # Indicate failure
    
    mask_bgr_resized = cv2.resize(mask_bgr, img_size, interpolation=cv2.INTER_NEAREST)
    mask_rgb_resized = cv2.cvtColor(mask_bgr_resized, cv2.COLOR_BGR2RGB)

    # Create a single-channel label ID mask (values 0 to num_classes-1)
    label_id_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8) 
    
    print(f"Processing mask: {os.path.basename(mask_path)}")
    # unique_colors_rgb = np.unique(mask_rgb_resized.reshape(-1, mask_rgb_resized.shape[2]), axis=0)
    # print(f"  Original unique RGB colors (after resize): {unique_colors_rgb.tolist()}")

    for class_id, color_rgb in enumerate(palette):
        matches = np.all(mask_rgb_resized == np.array(color_rgb, dtype=np.uint8), axis=2)
        label_id_mask[matches] = class_id
        # if np.sum(matches) > 0:
        #      print(f"    Mapping color {color_rgb} to class_id {class_id} for {np.sum(matches)} pixels.")

    # unique_ids_new = np.unique(label_id_mask)
    # print(f"  Generated label ID mask unique class IDs: {unique_ids_new.tolist()}")

    output_filename_base = os.path.basename(mask_path) # e.g., aachen_000000_000019_gtFine_color.png

    # Create and save binary masks for each class
    for i in range(num_classes):
        # Create binary mask for class i
        # Pixels belonging to class i are set to 255 (white), others to 0 (black)
        binary_mask_for_class_i = ((label_id_mask == i) * 255).astype(np.uint8)
        
        class_output_dir = os.path.join(output_base_mask_dir, str(i))
        ensure_dir(class_output_dir)
        
        output_path = os.path.join(class_output_dir, output_filename_base)
        cv2.imwrite(output_path, binary_mask_for_class_i)
        # if np.sum(binary_mask_for_class_i) > 0:
        #     print(f"    Saved binary mask for class {i} to {output_path}")
    
    return True # Indicate success

def main():
    parser = argparse.ArgumentParser(description='Preprocess Cityscape dataset.')
    parser.add_argument('--data_dir', type=str, default='inputs/cityscape',
                        help='Directory containing the Cityscape dataset (e.g., gtFine and leftImg8bit subdirs).')
    parser.add_argument('--output_dir', type=str, default='inputs/cityscape_224',
                        help='Output directory for preprocessed data.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size to resize images and masks to (square).')
    parser.add_argument('--img_subdir', type=str, default='image',
                        help='Subdirectory for images within data_dir (e.g., leftImg8bit/train).')
    parser.add_argument('--mask_subdir', type=str, default='mask',
                        help='Subdirectory for masks within data_dir (e.g., gtFine/train).')
    parser.add_argument('--img_suffix', type=str, default='_leftImg8bit.png',
                        help='Suffix for image files.')
    parser.add_argument('--mask_suffix', type=str, default='_leftImg8bit.png', # IMPORTANT: Use _color.png if input masks are colored
                        help='Suffix for mask files.')
    parser.add_argument('--test_run_limit', type=int, default=0,
                        help='Limit number of images to process for a test run (0 for all).')

    args = parser.parse_args()
    
    num_classes = len(CITYSCAPES_PALETTE) # Should be 20 based on your palette

    base_img_dir = os.path.join(args.data_dir, args.img_subdir)
    base_mask_dir = os.path.join(args.data_dir, args.mask_subdir)

    output_img_dir = os.path.join(args.output_dir, 'images')
    # This is the base directory for masks. Subdirectories 0, 1, ..., 19 will be created inside this.
    output_base_mask_dir = os.path.join(args.output_dir, 'masks')

    ensure_dir(output_img_dir)
    ensure_dir(output_base_mask_dir) # Ensure the main 'masks' directory exists

    img_paths = glob(os.path.join(base_img_dir, '**', f'*{args.img_suffix}'), recursive=True)
    
    if not img_paths:
        print(f"No images found in {base_img_dir} with suffix {args.img_suffix}. Please check the directory and suffix.")
        print(f"Glob pattern used: {os.path.join(base_img_dir, '**', f'*{args.img_suffix}')}")
        return
    
    print(f"Found {len(img_paths)} images.")

    img_ids_map = {}
    for p in img_paths:
        base_name = os.path.basename(p).replace(args.img_suffix, '')
        img_ids_map[base_name] = p
    
    img_ids_list = list(img_ids_map.keys())

    if not img_ids_list:
        print(f"No image IDs extracted. Please check the image files in {base_img_dir}.")
        return

    if args.test_run_limit > 0:
        img_ids_to_process = img_ids_list[:args.test_run_limit]
        print(f"Preprocessing first {len(img_ids_to_process)} images for testing...")
    else:
        img_ids_to_process = img_ids_list
        print(f"Preprocessing all {len(img_ids_to_process)} images...")

    processed_count = 0
    failed_mask_count = 0
    for img_id_base in img_ids_to_process:
        img_path = img_ids_map[img_id_base]
        
        relative_path_from_img_subdir = os.path.relpath(img_path, base_img_dir)
        mask_filename = os.path.basename(relative_path_from_img_subdir).replace(args.img_suffix, args.mask_suffix)
        mask_path_relative_to_mask_subdir = os.path.join(os.path.dirname(relative_path_from_img_subdir), mask_filename)
        mask_path = os.path.join(base_mask_dir, mask_path_relative_to_mask_subdir)

        print(f"\nProcessing image: {img_path}")
        # print(f"Attempting to find mask: {mask_path}") # Already printed in preprocess_mask

        preprocess_image(img_path, output_img_dir, (args.img_size, args.img_size))
        
        if os.path.exists(mask_path):
            success = preprocess_mask(mask_path, output_base_mask_dir, 
                                      (args.img_size, args.img_size), 
                                      palette=CITYSCAPES_PALETTE, 
                                      num_classes=num_classes)
            if success:
                processed_count +=1
            else:
                failed_mask_count +=1
        else:
            print(f"Warning: Mask not found for {img_id_base} at {mask_path}")
            failed_mask_count +=1


    print(f"\nPreprocessing complete.")
    print(f"  Successfully processed images and generated binary masks for: {processed_count} images.")
    if failed_mask_count > 0:
        print(f"  Failed to process or find masks for: {failed_mask_count} images.")
    print(f"Preprocessed data saved to {args.output_dir}")

if __name__ == '__main__':
    main()