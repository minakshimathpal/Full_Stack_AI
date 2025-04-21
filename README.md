Assignment5_MLOPs_Basics
# 🧠 MNIST CNN Classifier with GitHub Actions & Tests

## 🎯 Objective

Build a compact CNN model for MNIST digit classification that meets the following criteria:

- ✅ **Less than 25,000 parameters**
- ✅ **Achieves 95%+ training accuracy in just 1 epoch**
- ✅ **Includes CI/CD pipeline using GitHub Actions to verify model constraints**

---

## 📦 Features Implemented

### ✅ Base Assignment
- Built a compact CNN with <25K trainable parameters.
- Achieves **>95% accuracy in 1 epoch** using Adam optimizer.
- Configured **GitHub Actions** to automatically:
  - ✅ Test model parameter count
  - ✅ Test 1-epoch training accuracy
- Uploaded **screenshots and links** to GitHub Actions run and training logs.

### 🚀 Advanced Features
- 🔁 **Image Augmentation** using `torchvision.transforms`
- 🧪 **Additional CI Tests**:
  1. Test for training time under threshold
  2. Test for correct model architecture
  3. Test for dataset integrity (image shape & labels)
- 🟢 **Build Pass Badge** added to README

---

## 🧪 GitHub Actions Setup

### 📁 `.github/workflows/test_model.yml`

Includes steps to:
- Set up environment and dependencies
- Train model for 1 epoch
- Assert that:
  - `model.parameters_count < 25,000`
  - `training_accuracy >= 95%`
  - Additional custom tests pass

---

## 💻 Tech Stack

- **Language**: Python 3.8+
- **Libraries**: PyTorch, TorchVision, NumPy
- **Automation**: GitHub Actions
- **Testing**: PyTest + Custom Python Scripts

---

## 📸 Submission Checklist

- ✅ GitHub Actions Test Screenshot  
- ✅ [GitHub Actions Run Link](https://github.com/yourusername/mnist-cnn-ci/actions)  
- ✅ [README.md](https://github.com/yourusername/mnist-cnn-ci/blob/main/README.md)  
- ✅ Augmented Sample Image Screenshot  
- ✅ Test Code for 3 Unique Validations

---

## 📛 Badge

![Build Status](https://github.com/yourusername/mnist-cnn-ci/actions/workflows/test_model.yml/badge.svg)

---

## ✨ Author

Minakshi Mathpal  
[GitHub](https://github.com/minakshimathpal) | [LinkedIn](https://www.linkedin.com/in/minakshi-mathpal-9b78b915b)



"I am README" 
 main
