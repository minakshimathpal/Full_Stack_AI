# MNIST MLOps Project

This project implements a lightweight CNN model for MNIST digit classification with MLOps practices. The model is designed to achieve >95% accuracy in one epoch while keeping parameters under 25,000.

## Project Structure

```

## CI/CD Pipeline
The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs tests to verify:
   - Model has <25K parameters
   - Achieves >95% accuracy
5. Uploads trained model as artifact

## Model Architecture
- Input: 28x28 MNIST images
- 2 Convolutional layers
- MaxPooling
- Fully connected layer
- Output: 10 classes

## Testing
Tests verify:
1. Model parameter count (<25K)
2. Input/output shapes
3. Model accuracy (>95%)

## License
MIT

## Author
[Your Name]