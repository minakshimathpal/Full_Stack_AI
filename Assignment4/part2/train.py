import torch
import torch.optim as optim
from models.cnn import DynamicCNN
from utils.data import load_mnist_data
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = DynamicCNN(
            model_config['num_conv_layers'],
            model_config['kernels_per_layer']
        ).to(self.device)
        
        self.optimizer = self._get_optimizer(
            model_config['optimizer'],
            model_config['learning_rate']
        )
        
        self.train_loader, self.test_loader = load_mnist_data(
            model_config['batch_size']
        )
        logger.info(f"Model initialized on device: {self.device}")

    def _get_optimizer(self, optimizer_name, lr):
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Log batch progress periodically
            if batch_idx % 100 == 0:
                logger.debug(f'Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}]\t'
                           f'Loss: {loss.item():.6f}')

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        # Log epoch results
        logger.info(f'Train Epoch: {epoch}')
        logger.info(f'Average Loss: {avg_loss:.6f}')
        logger.info(f'Accuracy: {accuracy:.6f} ({correct}/{total})')

        return float(avg_loss), float(accuracy)  # Ensure we return float values

    def validate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += torch.nn.functional.nll_loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = correct / total

        # Log validation results
        logger.info(f'Validation Results:')
        logger.info(f'Average Loss: {avg_loss:.6f}')
        logger.info(f'Accuracy: {accuracy:.6f} ({correct}/{total})')

        return float(avg_loss), float(accuracy)  # Ensure we return float values