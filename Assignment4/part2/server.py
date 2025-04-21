import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import threading
import queue
from utils.data import load_mnist_data
import logging
from train import ModelTrainer
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables to store model trainers and training status
model_trainers = {}
training_thread = None
stop_training = False

@app.route('/')
def index():
    logger.info("Index page requested")
    return render_template('index.html')

def train_models(config, sid):
    global stop_training
    logger.info(f"Starting training with config: {config}")
    try:
        # Initialize model trainers
        model_trainers['model1'] = ModelTrainer(config['model1'])
        model_trainers['model2'] = ModelTrainer(config['model2'])
        
        logger.info("Models initialized successfully")
        
        # Get a batch of test images for later use
        _, test_loader = load_mnist_data(batch_size=1)
        test_images = []
        test_labels = []
        for data, target in test_loader:
            if len(test_images) < 2:
                test_images.append(data)
                test_labels.append(target.item())
            else:
                break

        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            if stop_training:
                break

            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Train both models
            m1_train_loss, m1_train_acc = model_trainers['model1'].train_epoch(epoch)
            m2_train_loss, m2_train_acc = model_trainers['model2'].train_epoch(epoch)
            
            # Validate both models
            m1_val_loss, m1_val_acc = model_trainers['model1'].validate()
            m2_val_loss, m2_val_acc = model_trainers['model2'].validate()

            # Debug logging for Model 1
            logger.info(f"""
            Model 1 Detailed Metrics (Epoch {epoch + 1}):
            - Training Loss: {m1_train_loss:.6f}
            - Training Accuracy: {m1_train_acc:.6f}
            - Validation Loss: {m1_val_loss:.6f}
            - Validation Accuracy: {m1_val_acc:.6f}
            """)

            # Debug logging for Model 2
            logger.info(f"""
            Model 2 Detailed Metrics (Epoch {epoch + 1}):
            - Training Loss: {m2_train_loss:.6f}
            - Training Accuracy: {m2_train_acc:.6f}
            - Validation Loss: {m2_val_loss:.6f}
            - Validation Accuracy: {m2_val_acc:.6f}
            """)

            # Create update data
            update_data = {
                'epoch': epoch + 1,
                'model1': {
                    'train_loss': float(m1_train_loss),
                    'train_acc': float(m1_train_acc),
                    'val_loss': float(m1_val_loss),
                    'val_acc': float(m1_val_acc)
                },
                'model2': {
                    'train_loss': float(m2_train_loss),
                    'train_acc': float(m2_train_acc),
                    'val_loss': float(m2_val_loss),
                    'val_acc': float(m2_val_acc)
                }
            }

            # Log the exact data being emitted
            logger.info(f"Emitting update data: {update_data}")

            # Emit training progress
            socketio.emit('training_update', update_data, room=sid)

            # Verify emission
            logger.info("Update data emitted successfully")

        logger.info("Training completed successfully! Generating predictions...")

        # Generate predictions for test images
        images_b64 = []
        model1_preds = []
        model2_preds = []

        for idx, img in enumerate(test_images):
            logger.info(f"Processing test image {idx + 1}/2")
            # Convert tensor to image and encode in base64
            img_np = img.squeeze().numpy()
            plt.figure(figsize=(2, 2))
            plt.imshow(img_np, cmap='gray')
            plt.axis('off')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            images_b64.append(f'data:image/png;base64,{img_b64}')

            # Get predictions
            with torch.no_grad():
                img = img.to(model_trainers['model1'].device)
                model1_preds.append(model_trainers['model1'].model(img).argmax(dim=1).item())
                model2_preds.append(model_trainers['model2'].model(img).argmax(dim=1).item())

        logger.info("Predictions generated successfully!")
        logger.info(f"Model 1 predictions: {model1_preds}")
        logger.info(f"Model 2 predictions: {model2_preds}")
        logger.info(f"True labels: {test_labels}")

        # Emit completion status with predictions
        socketio.emit('training_complete', {
            'images': images_b64,
            'true_labels': test_labels,
            'model1_preds': model1_preds,
            'model2_preds': model2_preds
        }, room=sid)

        logger.info("Training process completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.error(f"Error traceback:", exc_info=True)  # This will log the full traceback
        socketio.emit('error', {'message': str(e)}, room=sid)
    finally:
        stop_training = False

@socketio.on('start_training')
def handle_training_start(config):
    global training_thread, stop_training
    
    # Stop any existing training
    if training_thread and training_thread.is_alive():
        stop_training = True
        training_thread.join()
    
    # Start new training thread
    stop_training = False
    training_thread = threading.Thread(
        target=train_models,
        args=(config, request.sid)
    )
    training_thread.start()

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    global stop_training
    logger.info(f"Client disconnected: {request.sid}")
    stop_training = True

if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server...")
    socketio.run(app, debug=True, host='127.0.0.1', port=8000) 