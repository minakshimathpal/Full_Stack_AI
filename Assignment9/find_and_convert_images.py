import os
from PIL import Image
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def process_single_directory(args):
    """Helper function for parallel processing"""
    directory, class_id, convert, prefix = args
    class_dir = os.path.join(directory, class_id)
    non_rgb_files = []
    
    if not os.path.isdir(class_dir):
        return non_rgb_files
        
    for img_file in os.listdir(class_dir):
        if img_file.endswith('.JPEG'):
            img_path = os.path.join(class_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        non_rgb_files.append((img_path, img.mode))
                        if convert:
                            rgb_img = img.convert('RGB')
                            rgb_img.save(img_path, 'JPEG')
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
    
    return non_rgb_files

def process_images(data_dir, convert=False, workers=1):
    """
    Find non-RGB images and optionally convert them to RGB
    Args:
        data_dir: Path to ILSVRC subset directory
        convert: If True, will convert non-RGB images to RGB
        workers: Number of worker processes to use
    """
    train_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'train')
    val_dir = os.path.join(data_dir, 'Data', 'CLS-LOC', 'val')
    
    non_rgb_files = []
    
    for directory, prefix in [(train_dir, "train"), (val_dir, "val")]:
        print(f"\nProcessing {prefix} directory...")
        class_ids = os.listdir(directory)
        
        # Prepare arguments for parallel processing
        process_args = [(directory, class_id, convert, prefix) for class_id in class_ids]
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(process_single_directory, process_args),
                total=len(process_args),
                desc=f"Processing {prefix} directory"
            ))
            
            # Combine results from all workers
            for result in results:
                non_rgb_files.extend(result)

    # Report findings
    if non_rgb_files:
        print("\n\nFound non-RGB files:")
        for file_path, mode in non_rgb_files:
            print(f"\nFile: {file_path}")
            print(f"Mode: {mode}")
            if convert:
                print("✓ Converted to RGB")
    else:
        print("\n\nAll images are already in RGB format!")
    
    return len(non_rgb_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and convert non-RGB images in dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ILSVRC subset directory')
    parser.add_argument('--convert', action='store_true',
                      help='Convert non-RGB images to RGB format')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of worker processes to use')
    
    args = parser.parse_args()
    
    count = process_images(args.data_dir, args.convert, args.workers)
    
    if count > 0:
        print(f"\nFound {count} non-RGB files")
        if not args.convert:
            print("Run with --convert flag to convert images to RGB")
    else:
        print("\nAll images are in RGB format!") 