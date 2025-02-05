
# Neural Network Classification in C++

In this project, I built a simple feedforward neural network in C++ for classification tasks, completely from scratch without relying on frameworks like Torch or Tensorflow. The program handles data normalization, label encoding, training (via gradient descent with L2 regularization), evaluation using confusion matrices and per-class metrics, and finally, saving trained models in JSON format.

## Table of Contents


- [Project Structure](#project-structure)
- [Usage](#usage)
- [Compilation and Execution](#compilation-and-execution)


## Project Structure

### ActivationFunction Enumeration

Activation functions available are wrapped in an Enumeration(enum).


### LabelEncoder Class

- **Attributes:**
  - `label_to_index`: A map that converts string labels to unique integers.
  - `index_to_label`: A reverse map that helps recover the original string labels.
- **Functionality:**
  - `fit(const vector<string>& labels)`: to assign a unique number to each label.
  - `num_classes() const`: total number of unique classes.

### NeuralNetwork Class

This class is responsible for constructing, training, predicting, and saving my neural network model.

- **Constructor:**
  - I initialize the network’s weights and biases with random values, using a utilizable variant of He initialization to start in balanced conditions.
- **Key Methods:**
  - `forward(const vector<double>& input) const`:  processes input through the hidden layer, and then applies softmax to get probabilities.
  - `predict(const vector<double>& input) const`: returns index of the class with the highest probability.
  - `train(const vector<vector<double>>& features, const vector<int>& labels, int epochs)`: train the network for a number of epochs incl. fine-tuning the weights via gradient descent.
  - `saveModel(const string& filename)`: save model's settings and parameters (weights, biases, hyperparameters) to a JSON file.

- `normalize(vector<vector<double>>& data)`: to scale each feature to a range between 0 and 1, enhancing the training performance.

- `parseCSV(const string& filename, vector<vector<double>>& features, vector<int>& labels, LabelEncoder& label_encoder, int label_col, bool has_header)` : reads and parses a CSV file to extract features and labels, then converts the labels into numeric values using the `LabelEncoder`.

- `evaluate(const NeuralNetwork& nn, const vector<vector<double>>& features, const vector<int>& labels, const LabelEncoder& label_encoder)` : to assess model performance, print a confusion matrix, calculating overall accuracy, and providing per-class metrics like precision, recall, and F1-score.

## Usage

1. **Data Preprocessing in advance required:**  
  Data should be in a CSV format. Information on which column contains the labels and weather the CSV file includes a header row must also be known.

2. **Running the Program:**  
   When ran the executable, user asked on:
   - The path to dataset.
   - The column index where the labels are stored.
   - Whether the CSV file has a header.
   - The learning rate and number of training epochs.
   - The choice of activation function (ReLU, Sigmoid, Tanh, or Linear).
   - The number of neurons for the hidden layer.
   After training, evaluation metrics are shown and then user asked on saving the trained model in JSON format.

## Compilation and Execution

Compile the project using a C++ compiler (e.g., `g++`) and run the executable. For example:

```bash
g++ -std=c++11 -o neural_net NeuralNet.cpp
./neural_net
```
