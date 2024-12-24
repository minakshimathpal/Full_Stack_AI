# MNIST Classification with PyTorch

This project implements a CNN model for MNIST digit classification using PyTorch, focusing on efficient model architecture and modern training practices.

## Model Architecture
- Convolutional Neural Network (CNN)
- Uses Batch Normalization for better training stability
- Implements Dropout layers for regularization
- Global Average Pooling (GAP) and Fully Connected layers for classification
- Total Parameters: 7,788

## Key Features
- Achieves 99.4% test accuracy
- OneCycleLR scheduler for optimal learning rate management
- L1 regularization for model sparsity
- Comprehensive training visualization with loss and accuracy plots

## Project Structure
```bash
project/
├── src/
│ ├── model.py # Model architecture definition
│ ├── data_loader.py # Data loading and preprocessing
│ └── test.ipynb # Training and evaluation notebook
| ├── artifacts  # To save artifacts
│ ├── models  # folder to save trained models
│ └── test.ipynb # Training and evaluation notebook
| └── data  # Folder to save downloaded data
├── tests/│ 
│ └── test_model.py # Model architecture tests
├── conftest.py # Pytest configuration
|__README.md 
|__requirements.txt # To install dependencies
|__.github/workflows/ml-pipeline.yml # To run the pipeline
```

## Requirements
- Python 3.10
- PyTorch
- torchinfo
- matplotlib
- tqdm
- pytest (for testing)

## Model Performance
- Training Accuracy: 99.53%
- Test Accuracy: 99.42%
- Training Time: ~20 epochs

## Training Visualization
![Training Plots](Assignment6/src/artifacts/training_plots.png?v=1)
## Tests
The project includes automated tests to verify:
- Minimum parameter count (>10k parameters)
- Presence of Batch Normalization
- Implementation of Dropout
- Use of GAP or FC layers

Run tests using:
```bash
pytest tests/test_model.py -v -s
```

## Usage
1. Clone the repository
2. Install dependencies
3. Run the training notebook:
```bash
python src/test.ipynb
```

## Model Architecture Details
```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5            [-1, 8, 24, 24]             576
              ReLU-6            [-1, 8, 24, 24]               0
       BatchNorm2d-7            [-1, 8, 24, 24]              16
           Dropout-8            [-1, 8, 24, 24]               0
            Conv2d-9            [-1, 8, 24, 24]              64
        MaxPool2d-10            [-1, 8, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             720
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           1,440
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
AdaptiveAvgPool2d-27             [-1, 16, 1, 1]               0
          Dropout-28                   [-1, 16]               0
           Linear-29                   [-1, 10]             160
================================================================
Total params: 7,788
Trainable params: 7,788
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.45
Params size (MB): 0.03
Estimated Total Size (MB): 0.48
----------------------------------------------------------------
```
## Model Architecture Tests

The project includes a comprehensive test suite (`test_model.py`) to ensure the model meets architectural requirements.

### Test Coverage
```bash
tests/test_model.py
├── test_parameter_count() # Verifies model has 7-10k parameters
├── test_batch_normalization() # Checks for BatchNorm2d layers
├── test_dropout() # Ensures Dropout layers are present
└── test_gap_or_fc() # Validates GAP/FC layer implementation
```

### Running Tests
Run all tests with verbose output
```bash
pytest tests/test_model.py -v
```
Run tests with print statements
```bash
pytest tests/test_model.py -v -s
```
Run a specific test
```bash
pytest tests/test_model.py -v -k "test_parameter_count"
```

### Test Requirements
The model must satisfy these architectural constraints:
- Parameter count: 7,000-10,000 parameters
- Regularization: Must use Batch Normalization layers
- Dropout: Must implement dropout layers for regularization
- Output layer: Must use either Global Average Pooling or Fully Connected layer

### Sample Test Output

```bash
tests/test_model.py::test_parameter_count PASSED
tests/test_model.py::test_batch_normalization PASSED
tests/test_model.py::test_dropout PASSED
tests/test_model.py::test_gap_or_fc PASSED
```

### Adding New Tests
To add new architectural tests:
1. Add test functions to `tests/test_model.py`
2. Follow the naming convention `test_*`
3. Use pytest fixtures for common setup
4. Include descriptive assertions
