# emnist-character-classification
Comparison of MLP and CNN models on EMNIST dataset using PyTorch
# EMNIST Character Classification Using MLP and CNN

This project compares two deep learning models a **Multi-Layer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)** on the **EMNIST dataset**, which contains handwritten digits and characters. The goal is to classify these characters accurately using PyTorch, applying techniques like dropout, batch normalization, learning rate scheduling, and regularization.

---

## Dataset: EMNIST

- **Type:** Grayscale images of handwritten characters
- **Image size:** 28x28 pixels
- **Classes:** 47 balanced classes (digits 0-9 + uppercase/lowercase letters)
- **Samples:** 131,600 total (112,800 train, 18,800 test)
  
---

## Model Architectures

### MLP (Multi-Layer Perceptron)
- Input: Flattened 784-dimensional vector
- Hidden layers: (256, 128)
- Output: 47-class softmax layer
- Simple and fast, but limited for image tasks

### CNN (Convolutional Neural Network)
- Conv layers: [32 filters → MaxPooling] → [64 filters → MaxPooling]
- Fully connected layer: 128 units
- Output: 47-class softmax layer
- Better at detecting spatial patterns in images

---

## Techniques Used

| Category            | Methods Used                                   |
|---------------------|------------------------------------------------|
| **Activations**     | ReLU, LeakyReLU, ELU                           |
| **Optimizers**      | Adam, SGD, RMSprop                             |
| **Regularization**  | L1, L2 (optional), Dropout                     |
| **Batch Normalization** | Applied to stabilize training              |
| **Schedulers**      | ReduceLROnPlateau, StepLR                      |

---

## Evaluation Metrics

- **Training Loss**
- **Testing Loss**
- **Test Accuracy**
- **Predicted vs Actual Labels**

---
![image](https://github.com/user-attachments/assets/3269bb5e-b8ed-456d-9d1e-c959f8cbc378)

## Final Results

| Model | Test Accuracy |
|-------|---------------|
| MLP   | 84.99%        |
| CNN   | 86.00%        |

> The CNN slightly outperforms the MLP due to its ability to capture spatial features in image data.

## Key Learnings

- CNNs outperform MLPs for image-based tasks due to their ability to retain spatial information.
- Hyperparameter tuning plays a crucial role in improving model performance.
- Dropout, batch normalization, and learning rate scheduling help in reducing overfitting and accelerating convergence.
- Model evaluation using loss and accuracy metrics is essential for monitoring training quality.

---
