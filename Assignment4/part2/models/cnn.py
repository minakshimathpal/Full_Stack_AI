import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCNN(nn.Module):
    def __init__(self, num_conv_layers, kernels_per_layer):
        super(DynamicCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        # Input channels for MNIST is 1
        in_channels = 1
        
        # Create Conv2D layers dynamically
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels, kernels_per_layer[i], kernel_size=3, padding=1)
            )
            in_channels = kernels_per_layer[i]
        
        # Calculate final feature size after convolutions
        feature_size = 28 // (2 ** num_conv_layers)  # Due to max pooling
        final_channels = kernels_per_layer[-1] if kernels_per_layer else 1
        
        self.fc1 = nn.Linear(final_channels * feature_size * feature_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply Conv2D layers with ReLU and MaxPool2D
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            x = F.max_pool2d(x, 2)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 