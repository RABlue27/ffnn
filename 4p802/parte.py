import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, momentum):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = output_size
        self.momentum = momentum
        
        # Initialize weights with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size1)
        self.weights2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.weights3 = np.random.randn(self.hidden_size2, self.hidden_size3)
        self.weights4 = np.random.randn(self.hidden_size3, self.output_size)
        
        # Initialize previous weight updates to zero
        self.prev_update_w1 = np.zeros_like(self.weights1)
        self.prev_update_w2 = np.zeros_like(self.weights2)
        self.prev_update_w3 = np.zeros_like(self.weights3)
        self.prev_update_w4 = np.zeros_like(self.weights4)

    def __str__(self):
        return ("TBD")
            
    def forward(self, X):
        # Compute the dot product of the input with the first set of weights, and apply the sigmoid function
        self.hidden1 = sigmoid(np.dot(X, self.weights1))
        
        # Compute the dot product of the first hidden layer with the second set of weights, and apply the sigmoid function
        self.hidden2 = sigmoid(np.dot(self.hidden1, self.weights2))
        
        # Compute the dot product of the second hidden layer with the third set of weights, and apply the sigmoid function
        self.hidden3 = sigmoid(np.dot(self.hidden2, self.weights3))
        
        # Compute the dot product of the third hidden layer with the output set of weights, and apply the sigmoid function
        output = sigmoid(np.dot(self.hidden3, self.weights4))
        return output
        
    def backward(self, X, y, output, learning_rate):
        # Compute the difference between the output and the true label
        output_error = y - output
        
        # Compute the derivative of the output layer and update weights with momentum
        output_delta = output_error * sigmoid_derivative(output)
        update_w4 = learning_rate * np.dot(self.hidden3.T, output_delta)
        self.weights4 += update_w4 + self.momentum * self.prev_update_w4
        self.prev_update_w4 = update_w4
        
        # Compute the derivative of the third hidden layer and update weights with momentum
        hidden3_error = np.dot(output_delta, self.weights4.T)
        hidden3_delta = hidden3_error * sigmoid_derivative(self.hidden3)
        update_w3 = learning_rate * np.dot(self.hidden2.T, hidden3_delta)
        self.weights3 += update_w3 + self.momentum * self.prev_update_w3
        self.prev_update_w3 = update_w3
        
        # Compute the derivative of the second hidden layer and update weights with momentum
        hidden2_error = np.dot(hidden3_delta, self.weights3.T)
        hidden2_delta = hidden2_error * sigmoid_derivative(self.hidden2)
        update_w2 = learning_rate * np.dot(self.hidden1.T, hidden2_delta)
        self.weights2 += update_w2 + self.momentum * self.prev_update_w2
        self.prev_update_w2 = update_w2
        
        # Compute the derivative of the first hidden layer and update weights with momentum
        hidden1_error = np



        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



# Load the data from a file
data = np.loadtxt('L30fft25.out', skiprows=1)

# Split the data into inputs and labels
X = data[:, 1:]
y = data[:, 0].reshape(-1, 1)

# Normalize the inputs
X = X / np.max(X)

# Split the data into three equal subgroups
n = len(X) // 3
X1, X2, X3 = X[:n], X[n:2*n], X[2*n:]
y1, y2, y3 = y[:n], y[n:2*n], y[2*n:]

# Define the hyperparameters of the network
input_size = X.shape[1]
hidden_size1 = 6
hidden_size2 = 6
hidden_size3 = 6

output_size = 1
num_epochs = 10000
learning_rate = 0.1
momentum = 0

# Create an instance of the NeuralNetwork class
nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size, momentum)

# Create a table to display the results
table = PrettyTable()
table.field_names = ["Training Set", "Test Set", "MSE"]

# Define different test sets
test_sets = [
    (np.concatenate([X2,X3]), np.concatenate([y2,y3]), 'B'),
    (np.concatenate([X1,X3]), np.concatenate([y1,y3]), 'A'),
    (np.concatenate([X1,X2]), np.concatenate([y1,y2]), 'C')
]

# Create a table to display the results
table = PrettyTable()
table.field_names = ["Training Set", "Test Set", "MSE"]

# Train the network on the three permutations of groups
for i in range(2):
    momentum = 2 if i == 0 else 0.9
    for X_train, y_train, label in [(X1, y1, 'AB'), (X2, y2, 'AC'), (X3, y3, 'BC')]:
        nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size, momentum)
        errors = []
        for i in range(num_epochs):
            output = nn.forward(X_train)
            error = np.mean((output - y_train) ** 2)
            nn.backward(X_train, y_train, output, learning_rate)
            errors.append(error)
            if i % (num_epochs / 10) == 0:
                print(f"Epoch {i}, MSE: {error}")
        normalized_errors = np.array(errors) / np.max(errors)
        
        # Test the network on each test set
        for X_test, y_test, test_label in test_sets:
            mse = np.mean((nn.forward(X_test) - y_test) ** 2)
            table.add_row([label, test_label, mse])
        table.add_row(['---', '---', '---'])  # Add a separator row

# Print the results table
print(table)



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

# test_network(nn, 10, '16.txt')

print(nn)
