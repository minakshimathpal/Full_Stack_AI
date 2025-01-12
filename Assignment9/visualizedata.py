import os
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

def visualize_imagenet_data(data_dir):
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Choose a random image
    random_index = random.randint(0, len(dataset) - 1)
    image, label = dataset[random_index]
    
    # Reverse normalization (if applicable) and convert to numpy for visualization
    image = image.permute(1, 2, 0).numpy()
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Class: {dataset.classes[label]}")
    plt.savefig('output.png')
    plt.show()  # Ensure to show the plot for visualization

if __name__ == "__main__":
    data_dir = "/home/ubuntu/data_subset/CLS-LOC/val_subset"  # Update this to your dataset path
    visualize_imagenet_data(data_dir)
