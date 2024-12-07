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

@pytest.mark.parametrize("accuracy_threshold", [95])
def test_model_accuracy(models_dir, accuracy_threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    import glob
    import os
    model_files = glob.glob(f'{models_dir}/*.pth')
    assert model_files, f"No model files found in directory: {models_dir}"
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Test data
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
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