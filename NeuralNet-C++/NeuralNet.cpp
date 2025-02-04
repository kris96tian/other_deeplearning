#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <set>
#include <iomanip>
#include <numeric>
using namespace std;

enum ActivationFunction 
{
    ReLU = 1,
    Sigmoid,
    Tanh,
    Linear
};

class LabelEncoder 
{
    public:
        map<string, int> label_to_index;
        map<int, string> index_to_label;
        
        void fit(const vector<string>& labels) {
            int index = 0;
            for (const auto& label : labels) {
                if (label_to_index.find(label) == label_to_index.end()) {
                    label_to_index[label] = index;
                    index_to_label[index] = label;
                    index++;
                }
            }
        }
        
        int num_classes() const { return index_to_label.size(); }
};

class NeuralNetwork 
{
    public:
        NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate, double lambda, ActivationFunction activation_function);
        vector<double> forward(const vector<double>& input) const;
        void train(const vector<vector<double>>& features, const vector<int>& labels, int epochs);
        void saveModel(const string& filename);
        int predict(const vector<double>& input) const;

    private:
        int input_size;
        int hidden_size;
        int output_size;
        double learning_rate;
        double lambda;
        ActivationFunction activation_function;

        vector<vector<double>> W1, W2;
        vector<double> b1, b2;

        double sigmoid(double x) const;
        double relu(double x) const;
        double tanh_activation(double x) const;
        double linear(double x) const;
        double activation(double x) const;
        vector<double> softmax(const vector<double>& x) const;
};

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate, double lambda, ActivationFunction activation_function)  : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate), lambda(lambda),  activation_function(activation_function) 
{
    srand(static_cast<unsigned>(time(0)));
    W1 = vector<vector<double>>(input_size, vector<double>(hidden_size));
    W2 = vector<vector<double>>(hidden_size, vector<double>(output_size));
    b1 = vector<double>(hidden_size);
    b2 = vector<double>(output_size);

    double range1 = sqrt(2.0 / input_size);
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            W1[i][j] = ((double)rand() / RAND_MAX) * 2 * range1 - range1;
        }
    }

    double range2 = sqrt(2.0 / hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            W2[i][j] = ((double)rand() / RAND_MAX) * 2 * range2 - range2;
        }
    }

    for (int j = 0; j < hidden_size; ++j) b1[j] = ((double)rand() / RAND_MAX) * 0.01;
    for (int j = 0; j < output_size; ++j) b2[j] = ((double)rand() / RAND_MAX) * 0.01;
}

double NeuralNetwork::sigmoid(double x) const { return 1 / (1 + exp(-x)); }
double NeuralNetwork::relu(double x) const { return x > 0 ? x : 0; }
double NeuralNetwork::tanh_activation(double x) const { return tanh(x); }
double NeuralNetwork::linear(double x) const { return x; }

double NeuralNetwork::activation(double x) const 
{
    switch (activation_function) {
        case ReLU: return relu(x);
        case Sigmoid: return sigmoid(x);
        case Tanh: return tanh_activation(x);
        case Linear: return linear(x);
        default: return x;
    }
}

vector<double> NeuralNetwork::softmax(const vector<double>& x) const 
{
    vector<double> result(x.size());
    double max_x = *max_element(x.begin(), x.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp(x[i] - max_x);
        sum_exp += result[i];
    }
    for (size_t i = 0; i < x.size(); ++i) result[i] /= sum_exp;
    return result;
}

vector<double> NeuralNetwork::forward(const vector<double>& input) const 
{
    vector<double> hidden_layer(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        hidden_layer[i] = 0;
        for (int j = 0; j < input_size; ++j) hidden_layer[i] += input[j] * W1[j][i];
        hidden_layer[i] += b1[i];
        hidden_layer[i] = activation(hidden_layer[i]);
    }

    vector<double> output(output_size);
    for (int i = 0; i < output_size; ++i) {
        output[i] = 0;
        for (int j = 0; j < hidden_size; ++j) output[i] += hidden_layer[j] * W2[j][i];
        output[i] += b2[i];
    }
    return softmax(output);
}

int NeuralNetwork::predict(const vector<double>& input) const 
{
    vector<double> output = forward(input);
    return distance(output.begin(), max_element(output.begin(), output.end()));
}

void NeuralNetwork::saveModel(const string& filename) 
{
    ofstream file(filename);
    file << "{\n";
    file << "  \"input_size\": " << input_size << ",\n";
    file << "  \"hidden_size\": " << hidden_size << ",\n";
    file << "  \"output_size\": " << output_size << ",\n";
    file << "  \"learning_rate\": " << learning_rate << ",\n";
    file << "  \"lambda\": " << lambda << ",\n";
    file << "  \"activation_function\": " << activation_function << ",\n";

    file << "  \"W1\": [\n";
    for (size_t i = 0; i < W1.size(); ++i) {
        file << "    [";
        for (size_t j = 0; j < W1[i].size(); ++j) {
            file << W1[i][j];
            if (j < W1[i].size() - 1) file << ", ";
        }
        file << "]" << (i < W1.size() - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    file << "  \"W2\": [\n";
    for (size_t i = 0; i < W2.size(); ++i) {
        file << "    [";
        for (size_t j = 0; j < W2[i].size(); ++j) {
            file << W2[i][j];
            if (j < W2[i].size() - 1) file << ", ";
        }
        file << "]" << (i < W2.size() - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    file << "  \"b1\": [\n";
    for (size_t i = 0; i < b1.size(); ++i) {
        file << "    " << b1[i] << (i < b1.size() - 1 ? "," : "") << "\n";
    }
    file << "  ],\n";

    file << "  \"b2\": [\n";
    for (size_t i = 0; i < b2.size(); ++i) {
        file << "    " << b2[i] << (i < b2.size() - 1 ? "," : "") << "\n";
    }
    file << "  ]\n";
    file << "}\n";
    file.close();
}

void normalize(vector<vector<double>>& data) 
{
    for (size_t j = 0; j < data[0].size(); ++j) {
        double min_val = data[0][j], max_val = data[0][j];
        for (size_t i = 0; i < data.size(); ++i) {
            min_val = min(min_val, data[i][j]);
            max_val = max(max_val, data[i][j]);
        }
        if (max_val != min_val) {
            for (size_t i = 0; i < data.size(); ++i) {
                data[i][j] = (data[i][j] - min_val) / (max_val - min_val);
            }
        }
    }
}

void NeuralNetwork::train(const vector<vector<double>>& features, const vector<int>& labels, int epochs) 
{
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < features.size(); ++i) {
            vector<double> hidden_layer(hidden_size);
            for (int j = 0; j < hidden_size; ++j) {
                hidden_layer[j] = 0;
                for (int k = 0; k < input_size; ++k) {
                    hidden_layer[j] += features[i][k] * W1[k][j];
                }
                hidden_layer[j] += b1[j];
                hidden_layer[j] = activation(hidden_layer[j]);
            }

            vector<double> output(output_size);
            for (int j = 0; j < output_size; ++j) {
                output[j] = 0;
                for (int k = 0; k < hidden_size; ++k) {
                    output[j] += hidden_layer[k] * W2[k][j];
                }
                output[j] += b2[j];
            }
            output = softmax(output);

            int correct_label = labels[i];
            double loss = -log(output[correct_label]);
            total_loss += loss;

            vector<double> d_output(output_size, 0.0);
            d_output[correct_label] = -1.0 / output[correct_label];

            vector<vector<double>> dW2(hidden_size, vector<double>(output_size, 0.0));
            vector<double> db2(output_size, 0.0);
            for (int j = 0; j < output_size; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    dW2[k][j] = hidden_layer[k] * d_output[j];
                }
                db2[j] = d_output[j];
            }

            vector<double> d_hidden(hidden_size, 0.0);
            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < output_size; ++k) {
                    d_hidden[j] += W2[j][k] * d_output[k];
                }
                d_hidden[j] *= (hidden_layer[j] > 0) ? 1 : 0;
            }

            vector<vector<double>> dW1(input_size, vector<double>(hidden_size, 0.0));
            vector<double> db1(hidden_size, 0.0);
            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < input_size; ++k) {
                    dW1[k][j] = features[i][k] * d_hidden[j];
                }
                db1[j] = d_hidden[j];
            }

            for (int j = 0; j < input_size; ++j) {
                for (int k = 0; k < hidden_size; ++k) {
                    W1[j][k] -= learning_rate * (dW1[j][k] + lambda * W1[j][k]);
                }
            }

            for (int j = 0; j < hidden_size; ++j) {
                b1[j] -= learning_rate * db1[j];
            }

            for (int j = 0; j < hidden_size; ++j) {
                for (int k = 0; k < output_size; ++k) {
                    W2[j][k] -= learning_rate * (dW2[j][k] + lambda * W2[j][k]);
                }
            }

            for (int j = 0; j < output_size; ++j) {
                b2[j] -= learning_rate * db2[j];
            }
        }

        if (epoch % 10 == 0) {
            cout << "Epoch " << epoch << ", Loss: " << total_loss / features.size() << endl;
        }
    }
}

static void evaluate(const NeuralNetwork& nn, const vector<vector<double>>& features, const vector<int>& labels, const LabelEncoder& label_encoder)
{
        int num_classes = label_encoder.num_classes();
        vector<vector<int>> confusion_matrix(num_classes, vector<int>(num_classes, 0));
        int correct_predictions = 0;

        for (size_t i = 0; i < features.size(); ++i) {
            int predicted = nn.predict(features[i]);
            int actual = labels[i];
            
            if (actual >= 0 && actual < num_classes && predicted >= 0 && predicted < num_classes) {
                confusion_matrix[actual][predicted]++;
                if (predicted == actual) correct_predictions++;
            }
        }

        cout << "\nConfusion Matrix:" << endl;
        cout << "Predicted →" << endl;
        cout << "Actual ↓\t";
        for (int i = 0; i < num_classes; ++i) {
            cout << label_encoder.index_to_label.at(i) << "\t";
        }
        cout << endl;

        for (int i = 0; i < num_classes; ++i) {
            cout << label_encoder.index_to_label.at(i) << "\t";
            for (int j = 0; j < num_classes; ++j) {
                cout << confusion_matrix[i][j] << "\t";
            }
            cout << endl;
        }

        double accuracy = static_cast<double>(correct_predictions) / features.size();
        cout << fixed << setprecision(2);
        cout << "\nAccuracy: " << (accuracy * 100) << "%" << endl;

        cout << "\nPer-class metrics:" << endl;
        for (int i = 0; i < num_classes; ++i) {
            string class_name = label_encoder.index_to_label.at(i);
            
            int tp = confusion_matrix[i][i];
            int class_total = accumulate(confusion_matrix[i].begin(), confusion_matrix[i].end(), 0);
            int predicted_total = 0;
            for (int j = 0; j < num_classes; ++j) {
                predicted_total += confusion_matrix[j][i];
            }
            
            double precision = predicted_total > 0 ? tp / static_cast<double>(predicted_total) : 0;
            double recall = class_total > 0 ? tp / static_cast<double>(class_total) : 0;
            double f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
            
            cout << class_name << ":" << endl;
            cout << "  Precision: " << (precision * 100) << "%" << endl;
            cout << "  Recall:    " << (recall * 100) << "%" << endl;
            cout << "  F1-score:  " << (f1 * 100) << "%" << endl;
        }
    }


bool parseCSV(const string& filename, vector<vector<double>>& features, vector<int>& labels, LabelEncoder& label_encoder, int label_col, bool has_header) 
{
    ifstream file(filename);
    if (!file.is_open()) return false;

    vector<string> raw_labels;
    string line;
    bool first_line = true;

    while (getline(file, line)) {
        if (has_header && first_line) {
            first_line = false;
            continue;
        }

        vector<string> tokens;
        stringstream ss(line);
        string token;
        while (getline(ss, token, ',')) tokens.push_back(token);

        if (static_cast<int>(tokens.size()) <= label_col) return false;

        raw_labels.push_back(tokens[label_col]);
        
        vector<double> feature_row;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i == static_cast<size_t>(label_col)) continue;
            try {
                feature_row.push_back(stod(tokens[i]));
            } catch (...) {
                return false;
            }
        }
        features.push_back(feature_row);
    }

    label_encoder.fit(raw_labels);
    for (const auto& label : raw_labels) {
        labels.push_back(label_encoder.label_to_index.at(label));
    }

    file.close();
    return true;
}

int main() 
{
    vector<vector<double>> features;
    vector<int> labels;
    LabelEncoder label_encoder;
    string data_file, model_file;
    int hidden_size, epochs, activation_choice, label_col;
    double learning_rate, lambda = 0.01;
    bool has_header;

    cout << "Enter dataset path: ";
    cin >> data_file;
    cout << "Label column index: ";
    cin >> label_col;
    cout << "Has header (1/0): ";
    cin >> has_header;
    cout << "Learning rate: ";
    cin >> learning_rate;
    cout << "Epochs: ";
    cin >> epochs;
    cout << "Activation (1 ReLU,2 Sigmoid, 3 Tanh, 4 Linear ): ";
    cin >> activation_choice;
    cout << "Hidden layer size: ";
    cin >> hidden_size;

    if (!parseCSV(data_file, features, labels, label_encoder, label_col, has_header)) {
        cerr << "Error loading data" << endl;
        return 1;
    }

    normalize(features);

    int output_size = label_encoder.num_classes();
    NeuralNetwork nn(features[0].size(), hidden_size, output_size, 
                    learning_rate, lambda, static_cast<ActivationFunction>(activation_choice));
    nn.train(features, labels, epochs);
    
    evaluate(nn, features, labels, label_encoder);
    

    char save_choice;
    cout << "Save model? (y/n): ";
    cin >> save_choice;
    if (save_choice == 'y' || save_choice == 'Y') {
        cout << "Model filename: ";
        cin >> model_file;
        nn.saveModel(model_file + ".json");
    }

    return 0;
}
