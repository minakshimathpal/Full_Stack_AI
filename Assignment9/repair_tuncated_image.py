from PIL import Image
import os
import concurrent.futures
from tqdm import tqdm

def check_and_remove_or_convert_image(file_path):
    """
    Checks and processes a single image: removes corrupted images and converts non-RGB images to RGB.

    Args:
    file_path (str): The path of the image to be checked and processed.
    """
    try:
        # Open the image
        with Image.open(file_path) as img:
            # Convert image to RGB if not already in RGB mode
            if img.mode != 'RGB':
                print(f"Converting {file_path} to RGB")
                img = img.convert('RGB')
            img.verify()  # Verify if the image is valid
    except (IOError, SyntaxError) as e:
        print(f"Removing corrupted image: {file_path}")
        os.remove(file_path)  # Remove corrupted image

def repair_images_in_directory(data_dir):
    """
    Parallelizes the process of checking and repairing images in a given directory.

    Args:
    data_dir (str): The directory containing the images to be checked and processed.
    """
    # Create a list of all image files in the directory (recursively)
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            image_files.append(file_path)
    
    # Create a tqdm progress bar
    with tqdm(total=len(image_files), desc="Repairing images", ncols=100) as pbar:
        # Use concurrent.futures to parallelize the process
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Apply the check_and_remove_or_convert_image function to each image file
            # The progress bar is updated inside the lambda function
            for _ in executor.map(lambda file: check_and_remove_or_convert_image(file), image_files):
                pbar.update(1)

def repair_datasets(train_dir, val_dir):
    """
    Repair the training and validation datasets by removing corrupted images
    and converting non-RGB images to RGB.

    Args:
    train_dir (str): Path to the training dataset directory.
    val_dir (str): Path to the validation dataset directory.
    """
    print("Repairing and converting training dataset...")
    repair_images_in_directory(train_dir)

    print("Repairing and converting validation dataset...")
    repair_images_in_directory(val_dir)

# Set your directories here
train_dir = "/home/ubuntu/data_subset/CLS-LOC/train_subset"
val_dir = "/home/ubuntu/data_subset/CLS-LOC/val_subset"  # Update if needed

# Repair and convert datasets
repair_datasets(train_dir, val_dir)
