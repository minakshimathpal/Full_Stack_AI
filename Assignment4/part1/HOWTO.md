# MNIST Digit Recognition Web Application - Setup Guide

## Project Structure 
    project_root/
    ├── backend/
    │ ├── init.py
    │ ├── model/
    │ │ ├── init.py
    │ │ └── mnist_model.py
    │ ├── server.py
    │ ├── train.py
    │ └── logs/
    │
    ├── frontend/
    │ ├── static/
    │ │ └── css/
    │ │ └── style.css
    │ └── templates/
    │ └── index.html
    │
    └── README.md

## Prerequisites
- Python 3.8 or higher
- PyTorch
- Flask
- torchvision
- requests
- plotly.js (included via CDN)

### 1. Install Required Python Packages
```bash
pip install torch torchvision flask requests
```

## Running the Application

2. Start the Flask Server:
```bash
cd backend
python server.py
```
The server will start on `http://localhost:5000`

3. In a new terminal, start the training:
```bash
cd backend
python train.py
```

4. Open your web browser and navigate to:
```
http://localhost:5000
```

## Features
- Real-time training visualization
- Interactive loss and accuracy plots
- Final results display with digit predictions
- Automatic updates every 2 seconds

## Troubleshooting

### Common Issues:

1. **Port Already in Use**
   - Error: "Address already in use"
   - Solution: Change the port in server.py or kill the process using the port

2. **Module Not Found**
   - Error: "No module named 'torch'"
   - Solution: Ensure all required packages are installed using pip

3. **CUDA Not Available**
   - Note: The application will run on CPU if CUDA is not available
   - Solution: Install CUDA toolkit if GPU acceleration is needed

4. **File Path Issues**
   - Error: "Template not found"
   - Solution: Verify the project structure and file paths match the expected layout

## Development Notes

- The application uses Flask's debug mode by default
- Training progress is logged in `backend/logs/`
- Model checkpoints are saved as `mnist_cnn.pth`
- The frontend automatically refreshes every 2 seconds
- The training runs for 3 epochs by default (configurable in train.py)

## Additional Configuration

To modify the training parameters, edit `backend/train.py`:
- Change `epochs` variable for different training duration
- Adjust `batch_size` in data loaders
- Modify learning rate in optimizer configuration

To change the update frequency, edit `frontend/templates/index.html`:
- Modify the `setInterval` value (default: 2000ms)

## Support

For issues and questions:
1. Check the logs in `backend/logs/`
2. Verify console output in browser developer tools
3. Ensure all prerequisites are properly installed