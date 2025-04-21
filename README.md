# CNN Training Visualization & Model Comparison Interface

## 📌 Overview

This project demonstrates a complete pipeline to train CNN models and visualize their training logs in real-time. It features both a **basic** implementation and an **advanced comparison tool** that allows users to compare two CNN configurations side-by-side.

---

## 🎯 Assignment Goals

### ✅ Base Assignment [200 pts]

- Replicate training using a CNN model with three convolutional layers using 16, 32, and 64 kernels respectively (3x3 kernel size).
- Final dense layer is configured with dimensions: `7x7x64 → 10 classes`.
- Build a simple **frontend interface** to:
  - Display training logs
  - Show real-time updates of accuracy and loss per epoch

---

### 🚀 Advanced Assignment [200 pts + Bonus]

- **Compare Runs**:
  - Users can input two different sets of kernel sizes (e.g., 16-32-64 vs 8-8-8).
  - Train both models sequentially.
  - Visualize their **training loss and accuracy curves overlapped** on a graph for comparison.

- **Additional Features (Bonus 100 pts each)**:
  - Allow user to switch **optimizer** (e.g., Adam ↔ SGD)
  - Allow user to modify **batch size**
  - Allow user to adjust **number of epochs**
  
> All of these are controllable via the frontend interface.

---

## 💻 Tech Stack

- **Backend**: Python, PyTorch
- **Frontend**: Streamlit (or Flask + Chart.js)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Video Recording**: [Link to Submission Video (Insert Here)]

---

## 📂 Folder Structure
```
├── model/ # CNN architectures for various kernel configs
├── training/ # Training loops, loss tracking
├── frontend/ # UI to show logs and model comparison
├── utils/ # Data loaders, graph plotting, etc.
├── main.py # Main training script
├── README.md # Project documentation
```

## 📊 Features Implemented

- ✅ Customizable model architecture
- ✅ Real-time graph plotting of training progress
- ✅ Side-by-side comparison of model performance
- ✅ Adjustable optimizer, batch size, and epoch count (BONUS)

---

## 📽️ Submission Instructions

- [ ] Upload short video showing training logs and frontend comparisons
- [ ] Fill Q&A section with the features you actually implemented
- [ ] Ensure all code and dependencies are in the repo



