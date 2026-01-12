# Handwritten-Digit-Recognition-Image-Classification-
Project Overview

This project presents an end-to-end handwritten digit recognition system built using classical machine learning techniques. The task is to classify grayscale images of handwritten digits (0–9) from the MNIST dataset by implementing and evaluating multiple traditional ML models.

The project emphasizes data preprocessing, model implementation, hyperparameter tuning, evaluation, and analytical reporting, without using neural networks or pre-trained models.

# Objective

Classify handwritten digit images into classes 0–9

Build a complete machine learning pipeline using classical ML algorithms

Preprocess and normalize image data

Evaluate model performance using accuracy and confusion matrices

Analyze misclassifications and suggest improvements

# Outcome

Predict the correct digit label (0–9) for each input image

Report evaluation metrics:

Overall accuracy per model

Confusion matrices highlighting correct and incorrect predictions

Visualize:

Sample digit images

Misclassified examples with predicted vs actual labels

Provide a concise analytical comparison of models

# Dataset Description

Dataset: MNIST (CSV format)

Source: Kaggle

Image Resolution: 28 × 28 pixels

Features: 784 grayscale pixel values per image

Pixel Range: 0–255

Data Format

Each row in the CSV file contains:

label → digit class (0–9)

pixel0 to pixel783 → flattened pixel intensities

Example:

5,0,0,12,...,0

# Data Loading and Exploration

Load the dataset using Pandas

Inspect:

Total number of samples

Class distribution across digits

Visualize 5–10 sample images by reshaping pixel vectors to 28×28

Check for missing or invalid values

# Data Preprocessing

Normalize pixel values to the range [0, 1]

Split the dataset into training and testing sets (80/20 split)

(Optional) Apply Principal Component Analysis (PCA) to reduce dimensionality and improve computational efficiency, especially for SVM and KNN

# Model Implementation

Three classical machine learning models are trained and evaluated using scikit-learn:

#1 K-Nearest Neighbors (KNN)

Tune the number of neighbors (k)

Evaluate performance on the test set

#2️ Support Vector Machine (SVM)

Use Linear or RBF kernel

Tune hyperparameters such as C and gamma

#3️ Decision Tree

Tune structural parameters:

max_depth

min_samples_split

# Model Evaluation

Compute accuracy for each model

Generate confusion matrices for detailed per-class analysis

Visualize 5–10 misclassified digit images

Discuss potential reasons for incorrect predictions (e.g., similar digit shapes, noise, pixel ambiguity)

# Flowchart

┌────────────────────┐
│   MNIST CSV Data   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Data Loading (CSV) │
│ Pandas / NumPy     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Data Exploration   │
│ • Class balance    │
│ • Sample images    │
│ • Missing values   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Preprocessing      │
│ • Normalize [0,1]  │
│ • Train/Test split │
│ • (Optional) PCA   │
└─────────┬──────────┘
          ↓
┌──────────────────────────────────────┐
│ Model Training (Parallel)            │
│ ┌────────┐ ┌────────┐ ┌────────────┐ │
│ │  KNN   │ │  SVM   │ │ Decision   │ │
│ │        │ │        │ │ Tree       │ │
│ └────────┘ └────────┘ └────────────┘ │
└─────────┬────────────────────────────┘
          ↓
┌────────────────────┐
│ Model Evaluation   │
│ • Accuracy         │
│ • Confusion Matrix │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Error Analysis     │
│ • Misclassified    │
│   digits           │
│ • Visual inspection│
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Model Comparison   │
│ & Insights         │
└────────────────────┘


# Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

scikit-learn

# Key Takeaways

This project demonstrates how classical machine learning models perform on image-based classification tasks. While simpler models such as Decision Trees struggle with high-dimensional pixel data, models like SVM and KNN—especially when combined with proper preprocessing—can achieve strong performance and provide valuable insights into data characteristics and model behavior.
