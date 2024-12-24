import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN
import torch.nn.utils.prune as prune
import os
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    num_params = count_parameters(model)
    assert num_params < 25000, f"Model has {num_params} parameters, should be less than 25000"

def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_single_image_prediction(get_latest_model, mnist_test_data):
    images, labels = next(iter(mnist_test_data))  # Get the first batch
    image = images[0].unsqueeze(0)  # Get the first image and add batch dimension
    label = labels[0].item()
    cnn_model = get_latest_model
    output = cnn_model(image)
    _, predicted_label = torch.max(output, 1)
    
    assert predicted_label.item() == label, f"Single image prediction failed. True: {label}, Predicted: {predicted_label.item()}"

@pytest.mark.parametrize("accuracy_threshold", [95])
def test_model_accuracy(mnist_test_data,get_latest_model, accuracy_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_latest_model
    # get test data      
    test_loader = mnist_test_data   
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > accuracy_threshold, f"Model accuracy is {accuracy}%, should be > {accuracy_threshold}%" 


@pytest.mark.parametrize("accuracy_threshold", [90])
def test_robustness_against_noise(get_latest_model, mnist_test_data, artifacts_dir, accuracy_threshold):
    """
    Test model's robustness against noisy inputs and save noisy image visualization.
    """
    model = get_latest_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Add Gaussian noise to a sample batch
    data_iter = iter(mnist_test_data)
    images, labels = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)

    noise = torch.randn_like(images) * 0.3  # Adding small noise
    noisy_images = images + noise
    
    with torch.no_grad():
        outputs = model(noisy_images)
        _, predictions = torch.max(outputs, 1)
        correct = (predictions == labels).sum().item()
    
    accuracy = 100 * correct / labels.size(0)
    assert accuracy > accuracy_threshold, f"Model robustness to noise is inadequate. Accuracy: {accuracy}%, should be > {accuracy_threshold}%"

    # Save noisy image visualization
    os.makedirs(artifacts_dir, exist_ok=True)  # Ensure artifacts folder exists
    filepath = os.path.join(artifacts_dir, "noisy_images_visualization.png")
    plt.figure(figsize=(10, 5))
    num_images = 5
    for i in range(num_images):
        # Original image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap="gray")
        plt.title(f"Original: {labels[i].item()}")
        plt.axis("off")
        
        # Noisy image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(noisy_images[i].squeeze().cpu().numpy(), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()  # Close the plot to free memory
    print(f"Noisy images visualization saved at: {filepath}")

