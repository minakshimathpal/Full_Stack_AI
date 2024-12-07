import torch
import pytest
from torchvision import datasets, transforms
from model.network import SimpleCNN
import torch.nn.utils.prune as prune
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    num_params = count_parameters(model)
    assert num_params < 15000, f"Model has {num_params} parameters, should be less than 15000"

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
    assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%" 