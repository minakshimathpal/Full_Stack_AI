# MNIST CNN Comparison Tool - How To Guide

This guide will help you set up and run the MNIST CNN comparison tool, which allows you to train and compare two different CNN architectures on the MNIST dataset in real-time.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git (optional)

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Clone or download the project:
```bash
git clone <repository-url>
cd mnist_cnn
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. You only need one terminal. Start the Flask server:
```bash
python server.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

The application uses WebSocket for real-time communication, so everything runs through the single Flask server. You don't need multiple terminals.

## Using the Application

### Configuring Models

For each model (Model 1 and Model 2), you can configure:

1. **Number of Conv2D Layers**: Choose between 1-5 layers
   - For each layer, specify the number of kernels (filters)
   - Default values increase by powers of 2 (16, 32, 64, etc.)

2. **Training Parameters**:
   - Batch Size: 32, 64, or 128
   - Optimizer: Adam or SGD
   - Learning Rate: Between 0.0001 and 0.1

### Training Process

1. Configure both models as desired
2. Click the "Train Models" button to start training
3. Watch real-time updates:
   - Training and validation accuracy plots
   - Training and validation loss plots
4. After training completes:
   - View two random test images
   - Compare predictions from both models
   - See true labels and model predictions

### Tips for Best Results

1. **Model Architecture**:
   - Start with 2-3 Conv2D layers
   - Use increasing number of kernels per layer
   - Recommended kernel progression: 16 → 32 → 64

2. **Training Parameters**:
   - Start with batch size 64
   - Use Adam optimizer with learning rate 0.001
   - Adjust learning rate if training is unstable

3. **Comparison Strategy**:
   - Keep most parameters same between models
   - Change one aspect at a time for meaningful comparisons
   - Try comparing different optimizers or layer configurations

## Troubleshooting

1. **If training is slow**:
   - Reduce batch size
   - Use fewer Conv2D layers
   - Check if CUDA is being utilized

2. **If training crashes**:
   - Reduce model complexity
   - Check available GPU memory
   - Ensure stable internet connection for WebSocket

3. **If plots don't update**:
   - Refresh the page
   - Check browser console for errors
   - Restart the Flask server

## Technical Details

- Backend: Flask + SocketIO
- Frontend: HTML5, CSS3, JavaScript
- Deep Learning: PyTorch
- Visualization: Plotly.js
- Dataset: MNIST (automatically downloaded)

## Resource Usage

- GPU Memory: ~1-2GB per model
- CPU Usage: Moderate
- Network: Light (WebSocket communication)
- Disk Space: ~100MB (including MNIST dataset)

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your License Here]