# Assignment 7 - Model Experimentation

## Project Structure
```
Assignment_7/
├── README.md
├── data/
├── model_1.ipynb
├── model_2.ipynb
├── model_3.ipynb
├── model_1_summary.txt
├── model_2_summary.txt
└── model_3_summary.txt
```

## Models Overview
This project contains three different model implementations:

## **Model 1** 
   ([Model_1.ipynb](Model_1.ipynb))
   - See model_1_summary for detailed architecture
```
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
                Conv2d-9            [-1, 8, 22, 22]             576
            MaxPool2d-10            [-1, 8, 11, 11]               0
            Conv2d-11             [-1, 10, 9, 9]             720
                ReLU-12             [-1, 10, 9, 9]               0
        BatchNorm2d-13             [-1, 10, 9, 9]              20
            Dropout-14             [-1, 10, 9, 9]               0
            Conv2d-15             [-1, 16, 7, 7]           1,440
                ReLU-16             [-1, 16, 7, 7]               0
        BatchNorm2d-17             [-1, 16, 7, 7]              32
            Dropout-18             [-1, 16, 7, 7]               0
            Conv2d-19             [-1, 16, 5, 5]           2,304
                ReLU-20             [-1, 16, 5, 5]               0
        BatchNorm2d-21             [-1, 16, 5, 5]              32
            Dropout-22             [-1, 16, 5, 5]               0
            MaxPool2d-23             [-1, 16, 2, 2]               0
            Conv2d-24             [-1, 10, 2, 2]           1,440
                ReLU-25             [-1, 10, 2, 2]               0
        BatchNorm2d-26             [-1, 10, 2, 2]              20
            Dropout-27             [-1, 10, 2, 2]               0
    AdaptiveAvgPool2d-28             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 7,264
    Trainable params: 7,264
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.41
    Params size (MB): 0.03
    Estimated Total Size (MB): 0.44
    ----------------------------------------------------------------
```  
## Target 
  1. Get the skeleton of the model right.
  2.Perform MaxPooling at RF=7
  3.Fix DropOut, add it to each layer
## Results:
   1. Parameters: **7,264**
   2. Learning Rate: **0.0015**
   3. Best Train Accuracy: **99.68**
   4. Best Test Accuracy: **99.31** **(14th Epoch)**

 ## Analysis: 
     Works!
     But we're not seeing 99.4 or more as often as we'd like. We can further improve it. 
     The model is not over-fitting at all. 
     Seeing image samples, we can see that we can add slight rotation.

### Training Logs Model 1    
![Training Logs Model 1](artifacts/Model_1_logs.png?v=1)
## **Model 2** 
    ([model_2.ipynb](model_2.ipynb))
    - See model_2_summary for detailed architecture
    ```
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
                Conv2d-9            [-1, 8, 22, 22]             576
            MaxPool2d-10            [-1, 8, 11, 11]               0
            Conv2d-11             [-1, 10, 9, 9]             720
                ReLU-12             [-1, 10, 9, 9]               0
        BatchNorm2d-13             [-1, 10, 9, 9]              20
            Dropout-14             [-1, 10, 9, 9]               0
            Conv2d-15             [-1, 16, 7, 7]           1,440
                ReLU-16             [-1, 16, 7, 7]               0
        BatchNorm2d-17             [-1, 16, 7, 7]              32
            Dropout-18             [-1, 16, 7, 7]               0
            Conv2d-19             [-1, 16, 5, 5]           2,304
                ReLU-20             [-1, 16, 5, 5]               0
        BatchNorm2d-21             [-1, 16, 5, 5]              32
            Dropout-22             [-1, 16, 5, 5]               0
            MaxPool2d-23             [-1, 16, 2, 2]               0
            Conv2d-24             [-1, 10, 2, 2]           1,440
                ReLU-25             [-1, 10, 2, 2]               0
        BatchNorm2d-26             [-1, 10, 2, 2]              20
            Dropout-27             [-1, 10, 2, 2]               0
    AdaptiveAvgPool2d-28             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 7,264
    Trainable params: 7,264
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.41
    Params size (MB): 0.03
    Estimated Total Size (MB): 0.44
    ----------------------------------------------------------------
    ```

 ## Target: 
  1. Model 1 is working well. We can try to add Random rotation and RandomAffine.
 ## Results:
  1. Parameters: **7,264**
  2. Learning Rate: **0.0015**
  3. Best Train Accuracy: **99.30**
  4. Best Test Accuracy: **99.52** (13th Epoch)
  5. Consistenly **99.4+** validation accuracy from 11th Epoch
 ## Analysis:
 The model working fine. There is very minimal difference between train and test accuracies. This proves that the model is stable and will generalize well. This is fine, as we know we have made our training data harder. 
 The test accuracy is also up, which means our test data had few images that had transformation difference w.r.t. train dataset

### Training Logs Model 2   
![Training Logs Model 2](artifacts/Model_2_logs.png?v=1)


## **Model 3** 
   ([model_3.ipynb](model_3.ipynb))
    - See [model_3_summary.txt](model_3_summary.txt) for detailed architecture
```
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 8, 26, 26]              72
                ReLU-2            [-1, 8, 26, 26]               0
        BatchNorm2d-3            [-1, 8, 26, 26]              16
            Dropout-4            [-1, 8, 26, 26]               0
                Conv2d-5           [-1, 10, 24, 24]             720
                ReLU-6           [-1, 10, 24, 24]               0
        BatchNorm2d-7           [-1, 10, 24, 24]              20
            Dropout-8           [-1, 10, 24, 24]               0
            MaxPool2d-9           [-1, 10, 12, 12]               0
            Conv2d-10           [-1, 12, 10, 10]           1,080
                ReLU-11           [-1, 12, 10, 10]               0
        BatchNorm2d-12           [-1, 12, 10, 10]              24
            Dropout-13           [-1, 12, 10, 10]               0
            Conv2d-14           [-1, 15, 10, 10]           1,620
                ReLU-15           [-1, 15, 10, 10]               0
        BatchNorm2d-16           [-1, 15, 10, 10]              30
            Dropout-17           [-1, 15, 10, 10]               0
            MaxPool2d-18             [-1, 15, 5, 5]               0
            Conv2d-19             [-1, 15, 3, 3]           2,025
                ReLU-20             [-1, 15, 3, 3]               0
        BatchNorm2d-21             [-1, 15, 3, 3]              30
            Dropout-22             [-1, 15, 3, 3]               0
    AdaptiveAvgPool2d-23             [-1, 15, 1, 1]               0
            Conv2d-24             [-1, 10, 1, 1]             160
    ================================================================
    Total params: 5,797
    Trainable params: 5,797
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.44
    Params size (MB): 0.02
    Estimated Total Size (MB): 0.47
    ----------------------------------------------------------------

```
## Target: 
  1. Model 3 is working well.
  2. We added Random rotation and RandomAffine.
  3. Make model lighter and play with learning rate to see if we can get better results or comparable results    
     from model 2.
## Results:
  1. Parameters: **7,264**
  2. Learning Rate: **0.003**
  2. Best Train Accuracy: **99.32**
  3. Best Test Accuracy: **99.48** (11th Epoch)
  4. Consistenly **99.4+** validation accuracy from 11th Epoch
## Analysis:
  1. Parameters reduced from **7,264** to **5,797**.
  2. The model working fine. There is very minimal difference between train and test accuracies. This proves that the model is stable and will generalize well. This is fine, as we know we have made our training data harder. 
  2. The test accuracy is also up, which means our test data had few images that had transformation difference w.r.t. train dataset
  3. The model is lighter and the learning rate is higher. This means that the model is learning faster and is more likely to overfit. 
  4. The model is not overfitting, as the train and test accuracies are very close. 
  5. The model is not underfitting, as the train and test accuracies are very close. 
  6. The model is stable, as the train and test accuracies are very close. 
  7. The model is consistent, as the train and test accuracies are very close and showing 99.4 + validation accuracy from 11th Epoch
  8. The model is generalizing well, as the train and test accuracies are very close. 

### Training Logs Model 3   
![Training Logs Model 3](artifacts/Model_3_logs.png?v=1)

## Environment Setup
This project uses PyTorch and runs in a Python virtual environment named `pytorch_env`.

## Data
The data files are stored in the `data/` directory.

## Running the Notebooks
1. Ensure you have the `pytorch_env` environment activated:
