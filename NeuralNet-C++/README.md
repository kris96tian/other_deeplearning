
# Neural Network Classification in C++

This project implements a simple feedforward neural network for classification tasks in C++ without using external frameworks (like Torch or Tensorflow). It includes data preprocessing, label encoding, training using gradient descent with L2 regularization, model evaluation, and model persistence in JSON format.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Compilation and Execution](#compilation-and-execution)

## Overview

The code implements a neural network with a single hidden layer. It supports multiple activation functions (ReLU, Sigmoid, Tanh, Linear) and achieves the following:

- Reads and normalizes CSV data.
- Encodes categorical labels into numeric indices.
- Trains the network with backpropagation through a user-specified number of epochs.
- Evaluates performance with a confusion matrix, overall accuracy, and per-class metrics.
- Saves the network model (weights and configuration) in JSON format.

## Project Structure

### ActivationFunction Enum

Defines the available activation functions.

### LabelEncoder Class

- **Attributes:**
  - `label_to_index`: Maps string labels to integers.
  - `index_to_label`: Reverse mapping from integers to string labels.
- **Methods:**
  - `fit(const vector<string>& labels)`: Assigns unique integers to each label.
  - `num_classes() const`: Returns the number of unique classes.

### NeuralNetwork Class

Handles network initialization, prediction, training, and model saving.

- **Constructor:**
  - Initializes weights (using a He-style distribution) and biases with random values.
- **Methods:**
  - `forward(const vector<double>& input) const`: Computes network output by processing the input through the hidden and output layers, applying the softmax function.
  - `predict(const vector<double>& input) const`: Returns the class index with the highest predicted probability.
  - `train(const vector<vector<double>>& features, const vector<int>& labels, int epochs)`: Trains the network using gradient descent with L2 regularization.
  - `saveModel(const string& filename)`: Saves model parameters (including weights, biases, and hyperparameters) to a JSON file.
- **Private Activation Functions:**
  - `sigmoid`, `relu`, `tanh_activation`, and `linear` to provide the selected activation behavior.
  - `activation(double x) const`: Chooses the correct activation based on configuration.
  - `softmax(const vector<double>& x) const`: Computes the softmax probabilities for classification.

### Helper Functions

- **Data Normalization:**
  - `normalize(vector<vector<double>>& data)`: Scales each feature to the [0, 1] range.
- **CSV Parsing:**
  - `parseCSV(const string& filename, vector<vector<double>>& features, vector<int>& labels, LabelEncoder& label_encoder, int label_col, bool has_header)`: Reads the CSV and extracts features and labels. It expects a specified label column and optionally skips the header row.
- **Evaluation:**
  - `evaluate(const NeuralNetwork& nn, const vector<vector<double>>& features, const vector<int>& labels, const LabelEncoder& label_encoder)`: Computes and displays the confusion matrix, accuracy, and per-class metrics (precision, recall, F1-score).

## Usage

1. **Prepare Your Dataset:**
   - Ensure the data is in CSV format.
   - Identify the label column index.
   - Indicate whether the CSV includes a header row.

2. **Run the Program:**
   - The `main()` function will prompt for:
     - Dataset path
     - Label column index
     - Whether the CSV has a header
     - Learning rate and number of epochs
     - Activation function (ReLU, Sigmoid, Tanh, Linear)
     - Hidden layer size
   - After training, the program displays evaluation metrics.
   - You are then offered the option to save the trained model to a JSON file.

## Compilation and Execution

Compile the project using a C++ compiler (e.g., `g++`) and run the executable. For example:

```bash
g++ -std=c++11 -o neural_net main.cpp
./neural_net
```

After compilation, the program will prompt for user inputs regarding:
1. Dataset file path.
2. Label column index.
3. Presence of a header row.
4. Learning rate.
5. Epoch count.
6. Activation function choice.
7. Hidden layer size.
8. Option to save the model (with a specified JSON filename).
