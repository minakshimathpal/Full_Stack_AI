import pytest
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.network import SimpleCNN
@pytest.fixture(scope="module") 
def models_dir():
    """Provide the directory where models are saved."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "models")

@pytest.fixture(scope="module") 
def artifacts_dir():
    """Provide the directory where artifacts are saved."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "artifacts")

@pytest.fixture(scope="module")
def mnist_test_data():
    """Fixture to load the MNIST test dataset."""
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1000,shuffle=False)
    return test_loader

@pytest.fixture(scope="module")
def get_latest_model(models_dir):
    """Fixture to load the pre-trained CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = SimpleCNN().to(device)    
    
    # fetch the latest model
    import glob
    import os
    model_files = glob.glob(f'{models_dir}/*.pth')
    assert model_files, f"No model files found in directory: {models_dir}"
    latest_model = max(model_files, key=os.path.getctime)
    cnn_model.load_state_dict(torch.load(latest_model))  
    cnn_model.eval()
    return cnn_model