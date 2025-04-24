   # CNN vs ResNet50: A Comparative Study on Image Classification

This repository presents a comparative analysis between two deep learning approaches— a custom-built Convolutional Neural Network (CNN) and a fine-tuned ResNet50 Transfer Learning model— for solving a five-class image classification task.

## 🧾 Overview
The dataset includes 1,738 labeled images spread across five categories:
- Product 1
- Product 2
- Product 3
- Product 4
- Background

## 📁 Dataset

A sample dataset is included in the repository for demonstration purposes.  
It contains a small subset of images from the original 5 classes (Product 1–4 and Background) to help test the models and code structure.

> For full-scale training, replace the sample with the complete dataset using the same folder structure.


The goal is to classify images accurately into one of these classes using two different model architectures and analyze their strengths and limitations.

## 🚀 Models Implemented

### 🔹 Custom CNN
- Three convolutional layers (32, 64, 128 filters)
- MaxPooling after each Conv layer
- Dense(128) with ReLU activation
- Dropout(0.5)
- Output layer with softmax activation

### 🔹 ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- Base layers frozen initially
- Custom head with GlobalAveragePooling, Dense(128), Dropout, and output layer
- Fine-tuned by unfreezing top 40 layers

## 🧪 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

| Metric                  | CNN      | ResNet50 (Fine-Tuned) |
|-------------------------|----------|------------------------|
| Training Accuracy       | 96.27%   | 94.88%                 |
| Validation Accuracy     | 70.75%   | 63.50%                 |
| Validation Loss         | 2.37     | 0.89                   |
| Recall (Product 1)      | 27%      | 37%                    |
| F1-Score (Weighted Avg) | 0.66     | 0.60                   |

## 🧰 Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Jupyter Notebooks

## 📁 Repository Structure
```
.
├── cnn_model.py                  # CNN architecture and training script
├── resnet_model.py               # ResNet50 transfer learning implementation
├── data_preprocessing.py         # Image loading, resizing, and augmentation
├── evaluation_metrics.py         # Accuracy, F1-score, confusion matrix
├── /images                       # Plots and confusion matrices
├── IEEE_Report.pdf               # Final research report in IEEE format
├── Final_Image_Classification_Report.tex  # Full LaTeX source
└── README.md
```

## 📄 Report
The final report (IEEE format) includes:
- Full methodology
- Results with figures and tables
- Model comparison and discussion
- Future work

## 📌 Highlights
- Custom CNN achieved higher raw accuracy but overfitted
- ResNet50 generalized better and handled class imbalance more effectively

## 🔮 Future Improvements
- Add class weighting or SMOTE to handle imbalance
- Experiment with EfficientNet or Vision Transformers
- Ensemble CNN and ResNet50 for hybrid performance

## 📬 Contact
For questions or collaboration:
**Your Name**  
MSc in Data Analytics, Dublin Business School  
📧 your.email@example.com
