import torch
from src.model import Net
from torchinfo import summary
import pytest

def test_parameter_count():
    """Test if model has at least 10k parameters"""
    model = Net()
    model_stats = summary(model, input_size=(1, 1, 28, 28), verbose=0)
    total_params = model_stats.total_params
    assert total_params <= 10000, f'Model has only {total_params} parameters, minimum 10000 required'

def test_batch_normalization():
    """Test if model uses batch normalization"""
    model = Net()
    batch_norm_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    print(f"\nFound {len(batch_norm_layers)} batch normalization layers:")
    for idx, layer in enumerate(batch_norm_layers):
        print(f"BatchNorm layer {idx + 1}: {layer}")
    
    has_batchnorm = len(batch_norm_layers) > 0
    assert has_batchnorm, 'Model must use Batch Normalization'

def test_dropout():
    """Test if model uses dropout"""
    model = Net()
    dropout_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    print(f"\nFound {len(dropout_layers)} dropout layers:")
    for idx, layer in enumerate(dropout_layers):
        print(f"Dropout layer {idx + 1}: {layer}")
    
    has_dropout = len(dropout_layers) > 0
    assert has_dropout, 'Model must use Dropout'

def test_gap_or_fc():
    """Test if model uses either Global Average Pooling or Fully Connected layer"""
    model = Net()
    gap_layers = [m for m in model.modules() if isinstance(m, torch.nn.AdaptiveAvgPool2d)]
    fc_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    
    print(f"\nFound {len(gap_layers)} GAP layers and {len(fc_layers)} FC layers:")
    for idx, layer in enumerate(gap_layers):
        print(f"GAP layer {idx + 1}: {layer}")
    for idx, layer in enumerate(fc_layers):
        print(f"FC layer {idx + 1}: {layer}")
    
    has_gap = len(gap_layers) > 0
    has_fc = len(fc_layers) > 0
    assert has_gap or has_fc, 'Model must use either Global Average Pooling or Fully Connected layer' 

