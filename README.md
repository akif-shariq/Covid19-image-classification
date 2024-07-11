# Covid19-image-classification
This project aims to classify X-ray images into three categories: Normal, Pneumonia, and COVID-19 using deep learning techniques.

# Introduction
With the outbreak of COVID-19, there is a growing need for automated systems to assist medical professionals in diagnosing patients based on X-ray images. This project explores the use of Convolutional Neural Networks (CNNs) to classify X-ray scans into categories relevant to respiratory illnesses.

# Dataset
The dataset used in this project consists of X-ray images sourced from multiple repositories and databases, including COVID-19, Pneumonia, and Normal cases. The images are preprocessed and split into training and validation sets to train and evaluate the model.

# Methodology
## Model Architecture
The classification model is built using TensorFlow/Keras, utilizing a CNN architecture. Key components include convolutional layers, max-pooling layers, dropout layers for regularization, and a final dense layer with softmax activation for classification.

## Training
The model is trained using an ImageDataGenerator for data augmentation, enhancing the model's ability to generalize. The training process involves optimizing categorical crossentropy loss with the Adam optimizer and monitoring metrics such as categorical accuracy and AUC.

# Evaluation
After training, the model's performance is evaluated using the following:

Plots: Graphs depicting training/validation accuracy and AUC over epochs are generated using Matplotlib.

Classification Report: A classification report detailing precision, recall, and F1-score for each class is generated using scikit-learn's classification_report function.

Confusion Matrix: The confusion matrix, showing the distribution of true positive, false positive, true negative, and false negative predictions, is computed and displayed using scikit-learn's confusion_matrix function.

# Dependencies
TensorFlow/Keras
Matplotlib
NumPy
scikit-learn (for metrics evaluation)
