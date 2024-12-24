// Initialize Socket.IO connection
const socket = io();

// Initialize Plotly graphs
const accuracyLayout = {
    title: 'Training & Validation Accuracy',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Accuracy', range: [0, 1] }
};

const lossLayout = {
    title: 'Training & Validation Loss',
    xaxis: { title: 'Epoch' },
    yaxis: { title: 'Loss' }
};

// Initialize empty plots
Plotly.newPlot('accuracy-plot', [
    { name: 'Model 1 Train', x: [], y: [] },
    { name: 'Model 1 Val', x: [], y: [] },
    { name: 'Model 2 Train', x: [], y: [] },
    { name: 'Model 2 Val', x: [], y: [] }
], accuracyLayout);

Plotly.newPlot('loss-plot', [
    { name: 'Model 1 Train', x: [], y: [] },
    { name: 'Model 1 Val', x: [], y: [] },
    { name: 'Model 2 Train', x: [], y: [] },
    { name: 'Model 2 Val', x: [], y: [] }
], lossLayout);

// Function to update kernel input fields based on number of layers
function updateKernelInputs(modelNum) {
    const numLayers = document.getElementById(`model${modelNum}-layers`).value;
    const kernelsDiv = document.getElementById(`model${modelNum}-kernels`);
    kernelsDiv.innerHTML = '';

    for (let i = 0; i < numLayers; i++) {
        const label = document.createElement('label');
        label.innerHTML = `Kernels in Layer ${i + 1}:
            <input type="number" 
                   id="model${modelNum}-kernel-${i}" 
                   value="${16 * (2 ** Math.min(i, 2))}" 
                   min="1" 
                   max="512">`;
        kernelsDiv.appendChild(label);
    }
}

// Initialize kernel inputs for both models
updateKernelInputs(1);
updateKernelInputs(2);

// Function to gather model configuration
function getModelConfig(modelNum) {
    const numLayers = parseInt(document.getElementById(`model${modelNum}-layers`).value);
    const kernels = [];
    
    for (let i = 0; i < numLayers; i++) {
        kernels.push(parseInt(document.getElementById(`model${modelNum}-kernel-${i}`).value));
    }

    return {
        num_conv_layers: numLayers,
        kernels_per_layer: kernels,
        batch_size: parseInt(document.getElementById(`model${modelNum}-batch-size`).value),
        optimizer: document.getElementById(`model${modelNum}-optimizer`).value,
        learning_rate: parseFloat(document.getElementById(`model${modelNum}-lr`).value)
    };
}

// Function to start training
function startTraining() {
    // Disable the train button
    document.getElementById('train-button').disabled = true;
    
    // Clear existing plots
    Plotly.purge('accuracy-plot');
    Plotly.purge('loss-plot');
    Plotly.newPlot('accuracy-plot', [
        { name: 'Model 1 Train', x: [], y: [] },
        { name: 'Model 1 Val', x: [], y: [] },
        { name: 'Model 2 Train', x: [], y: [] },
        { name: 'Model 2 Val', x: [], y: [] }
    ], accuracyLayout);
    Plotly.newPlot('loss-plot', [
        { name: 'Model 1 Train', x: [], y: [] },
        { name: 'Model 1 Val', x: [], y: [] },
        { name: 'Model 2 Train', x: [], y: [] },
        { name: 'Model 2 Val', x: [], y: [] }
    ], lossLayout);

    // Get configurations for both models
    const config = {
        model1: getModelConfig(1),
        model2: getModelConfig(2)
    };

    // Send configurations to server
    socket.emit('start_training', config);
}

// Socket event handlers
socket.on('training_update', function(data) {
    // Update accuracy plot
    Plotly.extendTraces('accuracy-plot', {
        y: [[data.model1.train_acc], [data.model1.val_acc], 
            [data.model2.train_acc], [data.model2.val_acc]],
        x: [[data.epoch], [data.epoch], [data.epoch], [data.epoch]]
    }, [0, 1, 2, 3]);

    // Update loss plot
    Plotly.extendTraces('loss-plot', {
        y: [[data.model1.train_loss], [data.model1.val_loss],
            [data.model2.train_loss], [data.model2.val_loss]],
        x: [[data.epoch], [data.epoch], [data.epoch], [data.epoch]]
    }, [0, 1, 2, 3]);
});

socket.on('training_complete', function(data) {
    // Enable the train button
    document.getElementById('train-button').disabled = false;
    
    // Update test images and predictions
    for (let i = 1; i <= 2; i++) {
        document.getElementById(`test-image-${i}`).src = data.images[i-1];
        document.getElementById(`true-label-${i}`).textContent = data.true_labels[i-1];
        document.getElementById(`model1-pred-${i}`).textContent = data.model1_preds[i-1];
        document.getElementById(`model2-pred-${i}`).textContent = data.model2_preds[i-1];
    }
});

socket.on('error', function(error) {
    alert('Error: ' + error.message);
    document.getElementById('train-button').disabled = false;
}); 