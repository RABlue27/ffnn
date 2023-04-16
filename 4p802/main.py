import numpy as np
from prettytable import PrettyTable



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)

    def __str__(self):
        return str(self.weights2)
            
        
    def forward(self, X):
        # Compute the dot product of the input with the first set of weights, and apply the sigmoid function
        self.hidden = sigmoid(np.dot(X, self.weights1))
        
        # Compute the dot product of the hidden layer with the second set of weights, and apply the sigmoid function
        output = sigmoid(np.dot(self.hidden, self.weights2))
        return output
        
    def backward(self, X, y, output, learning_rate):
        # Compute the difference between the output and the true label
        output_error = y - output
        
        # Compute the derivative of the output layer and update weights
        output_delta = output_error * sigmoid_derivative(output)
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_delta)
        
        # Compute the derivative of the hidden layer and update weights
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)
        
    def train(self, X, y, num_epochs, learning_rate):
        for i in range(num_epochs):
            # Compute the output of the network for the current input
            output = self.forward(X)

            # Compute the mean squared error of the output
            error = np.mean((output - y) ** 2)
        
            # Update the weights based on the output and true labels
            self.backward(X, y, output, learning_rate)
            
            # Print the mean squared error every 10 epochs
            if i % (num_epochs / 10) == 0:
                print(f"Epoch {i}, MSE: {error}")


        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



# Load the data from a file
data = np.loadtxt('L30fft1000.out', skiprows=1)

# Split the data into inputs and label as 
X = data[:, 1:]
y = data[:, 0].reshape(-1, 1)

# Normalize the inputs
X = X / np.max(X)

# Define the hyperparameters of the network
input_size = X.shape[1]
hidden_size = 24
output_size = 1
num_epochs = 10000
learning_rate = 0.1

# Create an instance of the NeuralNetwork class
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network on the data
nn.train(X, y, num_epochs, learning_rate)

def test_network(nn, num_inputs, data_file):
    # Load the data from the file
    data = np.loadtxt(data_file, skiprows=1)
    
    # Normalize the input data
    inputs = data[:, 1:] / np.max(data[:, 1:])
    
    # Generate random indices to test
    test_indices = np.random.choice(range(len(data)), size=num_inputs, replace=False)
    
    # Initialize counters for correct and total predictions
    num_correct = 0
    total_predictions = 0
    
    # Initialize the pretty table to display the results
    table = PrettyTable(['Test Input Index', 'True Label', 'Predicted Label', 'Result'])
    
    # Iterate over the selected inputs and test each one
    for i in range(num_inputs):
        input_idx = test_indices[i]
        true_label = int(data[input_idx, 0])
        input_data = inputs[input_idx]
        
        # Compute the output of the network for the input data
        output = nn.forward(input_data)
        predicted_label = int(round(output[0]))
        
        # Increment the correct and total prediction counters
        if predicted_label == true_label:
            num_correct += 1
            result = 'Match'
        else:
            result = 'Fail'
        total_predictions += 1
        
        # Add a row to the pretty table for this test result
        table.add_row([input_idx, true_label, predicted_label, result])
    
    # Print the pretty table to the console
    print(table)
    
    # Compute the success rate and print it to the console
    success_rate = num_correct / total_predictions
    print(f"Success rate: {success_rate}")

# test_network(nn, 10, 'L30fft1000.out')

print(nn)