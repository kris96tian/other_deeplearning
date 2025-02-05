# Neural Network Classification in C++

This project implements a simple feedforward neural network for classification tasks. The code includes components for data preprocessing, label encoding, training the network using gradient descent with regularization, model evaluation with confusion matrix and per-class metrics, and saving the trained model in JSON format, without using a framework like Torch or Tensorflow.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
  - [ActivationFunction Enum](#activationfunction-enum)
  - [LabelEncoder Class](#labelencoder-class)
  - [NeuralNetwork Class](#neuralnetwork-class)
  - [Helper Functions](#helper-functions)
- [Usage](#usage)
- [Compilation and Execution](#compilation-and-execution)

## Overview

The code provides a simple neural network with a single hidden layer. It supports several activation functions (ReLU, Sigmoid, Tanh, Linear) and performs the following tasks:
- Reads and normalizes data from a CSV file.
- Encodes categorical labels into numeric indices.
- Trains the network over a specified number of epochs using backpropagation.
- Evaluates the trained network by printing a confusion matrix, accuracy, and per-class metrics.
- Offers an option to save the trained model in JSON format.


## Project Structure

### ActivationFunction Enum


- **Purpose:** Defines the types of activation functions available for the neural network.


### LabelEncoder Class


- **Members:**
  - `label_to_index`: Maps each unique string label to an integer.
  - `index_to_label`: Reverse mapping from integer indices to string labels.
  
- **Methods:**
  - `fit(const vector<string>& labels)`: Iterates over the input labels and assigns a unique integer to each label.
  - `num_classes() const`: Returns the number of unique labels (i.e., classes).

### NeuralNetwork Class

#### Constructor

- **Purpose:** Initializes the neural network weights and biases with random values.
- **Weight Initialization:** Uses a random distribution scaled by the square root of the number of inputs to each layer (He initialization style for the first layer and similar for the second).

#### Private Activation Methods

- `sigmoid(double x) const`: Returns the sigmoid activation value.
- `relu(double x) const`: Returns the ReLU activation value.
- `tanh_activation(double x) const`: Returns the tanh activation value.
- `linear(double x) const`: Returns the linear (identity) activation value.
- `activation(double x) const`: Chooses and applies the appropriate activation function based on the `activation_function` member.
- `softmax(const vector<double>& x) const`: Computes the softmax over the output layer to obtain probabilities.

#### Forward Pass

```cpp
vector<double> NeuralNetwork::forward(const vector<double>& input) const;
```

- **Purpose:** Computes the output of the network for a given input.
- **Process:**
  - Computes the hidden layer activations using the weighted sum of inputs, adds biases, and applies the activation function.
  - Computes the output layer by taking the weighted sum of the hidden layer activations and adding biases.
  - Applies the softmax function to the output layer to generate probability distributions.

#### Prediction

```cpp
int NeuralNetwork::predict(const vector<double>& input) const;
```

- **Purpose:** Determines the class with the highest probability from the forward pass output.
- **Usage:** Returns the index corresponding to the predicted class.

#### Training

```cpp
void NeuralNetwork::train(const vector<vector<double>>& features, const vector<int>& labels, int epochs);
```

- **Purpose:** Trains the neural network using a simple gradient descent algorithm.
- **Key Steps in Training:**
  - **Forward Pass:** Compute activations for each sample.
  - **Loss Computation:** Uses cross-entropy loss calculated as `-log(probability of the correct class)`.
  - **Backward Pass:** Computes gradients for the weights and biases (for both layers) and applies L2 regularization.
  - **Parameter Update:** Adjusts weights and biases using the learning rate.
  - **Logging:** Every 10 epochs, the average loss is printed.

#### Model Saving

```cpp
void NeuralNetwork::saveModel(const string& filename);
```

- **Purpose:** Saves the network configuration, weights, and biases to a JSON file.
- **Format:** The output file contains model parameters like input size, hidden size, output size, learning rate, regularization lambda, activation function, and the actual network parameters (W1, W2, b1, b2).

### Helper Functions

#### Data Normalization

```cpp
void normalize(vector<vector<double>>& data);
```

- **Purpose:** Normalizes each feature column in the dataset to a [0, 1] range.
- **Process:** For each feature, computes the minimum and maximum values, then scales all entries accordingly.

#### CSV Parsing

```cpp
bool parseCSV(const string& filename, vector<vector<double>>& features, vector<int>& labels, LabelEncoder& label_encoder, int label_col, bool has_header);
```

- **Purpose:** Reads a CSV file and extracts features and labels.
- **Parameters:**
  - `filename`: Path to the CSV file.
  - `features`: Container for the numeric features.
  - `labels`: Container for the labels (which will later be encoded).
  - `label_encoder`: An instance of `LabelEncoder` that will encode the string labels.
  - `label_col`: Column index in the CSV file that contains the labels.
  - `has_header`: Indicates if the CSV file has a header row to skip.
- **Process:** Parses each line, extracts features (converting strings to `double`), and collects raw labels. After reading, it fits the label encoder and converts raw labels to indices.

#### Evaluation

```cpp
static void evaluate(const NeuralNetwork& nn, const vector<vector<double>>& features, const vector<int>& labels, const LabelEncoder& label_encoder);
```

- **Purpose:** Evaluates the performance of the neural network.
- **Outputs:**
  - **Confusion Matrix:** Displays the number of true positives, false positives, and false negatives per class.
  - **Overall Accuracy:** Computes and prints the accuracy as a percentage.
  - **Per-Class Metrics:** For each class, calculates and displays precision, recall, and F1-score.
- **Note:** The function uses the `LabelEncoder` to map between numeric indices and the original string labels.

## Usage

1. **Prepare your Dataset:**  
   - Ensure your data is in CSV format.
   - Specify which column contains the labels.
   - Optionally include a header row.

2. **Running the Program:**  
   - The `main()` function prompts the user for:
     - Dataset path.
     - Label column index.
     - Whether the CSV has a header.
     - Learning rate.
     - Number of epochs.
     - Choice of activation function (1: ReLU, 2: Sigmoid, 3: Tanh, 4: Linear).
     - Hidden layer size.
   - After training, the program evaluates the model and displays performance metrics.
   - The user is then prompted to save the model to a JSON file.

## Compilation and Execution

Compilation is done using a C++ compiler. For example, using `g++`:

```bash
g++ -std=c++11 -o neural_net main.cpp
```

And then run:

```bash
./neural_net
```

 **After compilation:**:
   1. User input for the path to the dataset file.
   2. User input for the column index of the label in the dataset.
   3. User input on weather the dataset has a header row. (y/n)
   4. User input for the learning rate.
   5. User input for the number of epochs.
   6. User input for the activation function. (enums)
   7. User input for the hidden layer size.
   8. User asked on saving the trained model. (if yes, user input for the model filename)

