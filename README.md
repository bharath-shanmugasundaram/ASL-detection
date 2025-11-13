# 🤟 American Sign Language (ASL) Alphabet Classification using AlexNet (PyTorch)

This project implements an **ASL Alphabet Recognition System** using a custom-designed **AlexNet-inspired Convolutional Neural Network** trained on the **ASL Alphabet Dataset**.  
The model predicts one of **29 possible classes**, including the alphabet letters A–Z and special signs such as **DEL**, **NOTHING**, and **SPACE**.

---

## 📌 Project Overview

The goal of this project is to build a deep learning system capable of recognizing hand gesture images representing American Sign Language letters.  
Key elements of the project include:

- Loading & organizing thousands of labeled ASL images
- Preprocessing, resizing, normalizing, and converting images to tensors
- Mapping 29 ASL labels to index values for training
- Building a custom **AlexNet-style CNN architecture**
- Training using PyTorch with GPU/Apple MPS acceleration
- Evaluating model performance on both train/test splits and external test data
- Generating a **confusion matrix** for detailed class-level visualization
- Saving the trained model for later use

---

## 📂 Dataset Structure

The training data follows this structure:



Each subfolder contains multiple gesture images belonging to that class.

The dataset contains **tens of thousands** of RGB images, making it suitable for deep CNNs.

---

## 🧠 Class Labels (29 Total)

The 29 classes used in the project:




Special classes:

- **DEL**  
- **SPA** (Space)  
- **NOT** (Nothing)

---

## 🔄 Preprocessing Pipeline

Each image is passed through a standard transform:

- Convert to PIL  
- Resize to **64 × 64**  
- Convert to Tensor  
- Normalize using mean = 0.5, std = 0.5  

This ensures all images are uniform and suitable for batch training.

---

## 🧱 Model Architecture — AlexNet Variant

A customized AlexNet-like model is used featuring:

### **Convolutional Layers**
- 3 convolution blocks with ReLU
- Max-pooling for downsampling
- Final feature map flattened to a huge vector

### **Fully Connected Layers**
- Two dense layers (4096 neurons each)
- Dropout for regularization
- Output layer with **29 units** (one per ASL class)

This architecture is powerful enough to extract spatial and gesture-level features from hand images.

---

## 🏋️ Model Training

Training details:

- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam (lr = 0.001)  
- **Batch size:** 64  
- **Epochs:** 10  
- **Device:** MPS (Mac GPU) or CPU  
- **Train/Test Split:** 80% / 20%

During each epoch, the script:

- Runs a standard training loop  
- Evaluates accuracy on the test set  
- Logs epoch loss & accuracy  

---

## 📈 Evaluation

Evaluation is performed in two ways:

### ✔️ Test-Loader Evaluation  
Measures accuracy on the validation/test set after every epoch.

### ✔️ External ASL Test Folder  
Each image is loaded individually and classified:


This simulates real-world usage.

### ✔️ Confusion Matrix  
A **normalized 29×29 confusion matrix** is plotted to visualize:

- Most confident predictions  
- Misclassified signs  
- Per-class performance  

---

## 🖼️ Visualization

A high-resolution confusion matrix is generated using:

- sklearn  
- matplotlib  
- ConfusionMatrixDisplay  

This visually explains how well the model distinguishes between different hand signs.

---

## 💾 Saving & Reusing the Model

The trained model weights are saved as:


This file can later be loaded for:

- Inference
- Fine-tuning
- Real-time ASL recognition projects

---

## 🛠️ Technologies Used

- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Scikit-Learn**
- **Matplotlib**
- **NumPy**

---

## 🚀 Future Improvements

- Integrate **real-time webcam detection**
- Use a stronger CNN like **ResNet**, **Inception**, or **EfficientNet**
- Apply data augmentation for more robustness
- Build a **Live ASL Translation App** with PyTorch + Streamlit

---

## 🙌 Acknowledgements

- ASL Alphabet Dataset creators  
- PyTorch team  
- OpenCV open-source contributors  

---

If you want this:

✅ Packaged into a **downloadable README.md**  
✅ Improved styling or emojis  
✅ Folder structure suggestion  
Just let me know!  
