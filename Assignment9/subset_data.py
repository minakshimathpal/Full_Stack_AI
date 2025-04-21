import os
import random
import shutil

# Paths to the ImageNet training and validation directories
TRAIN_DIR = '/home/ubuntu/raw_data/extracted/ILSVRC/Data/CLS-LOC/train'
VAL_DIR = '/home/ubuntu/raw_data/extracted/ILSVRC/Data/CLS-LOC/val'

# Paths to the new subset directories
SUBSET_TRAIN_DIR = '/home/ubuntu/Assignment9/data_subset/CLS-LOC/train_subset'
SUBSET_VAL_DIR = '/home/ubuntu/Assignment9/data_subset/CLS-LOC/val_subset'

# Function to subset a directory (select 10% of images from each class)
def subset_data(input_dir, output_dir, subset_ratio=0.1):
    # Loop over each class in the directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            print(f"Subsetting class: {class_name}")
            
            # Create the corresponding class directory in the subset
            class_output_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # Get a list of all files in the class directory
            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            total_files = len(files)
            subset_size = int(total_files * subset_ratio)

            # Randomly select 10% of the files
            selected_files = random.sample(files, subset_size)

            # Copy the selected files to the output directory
            for file in selected_files:
                shutil.copy(os.path.join(class_dir, file), os.path.join(class_output_dir, file))

# Create subset directories
os.makedirs(SUBSET_TRAIN_DIR, exist_ok=True)
os.makedirs(SUBSET_VAL_DIR, exist_ok=True)

# Subset training and validation data
print("Subsetting training data...")
subset_data(TRAIN_DIR, SUBSET_TRAIN_DIR)
print("Subsetting validation data...")
subset_data(VAL_DIR, SUBSET_VAL_DIR)

print("Subset creation completed successfully!")
