import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, momentum):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.momentum = momentum
        
        # Initialize weights with random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize previous weight updates to zero
        self.prev_update_w1 = np.zeros_like(self.weights1)
        self.prev_update_w2 = np.zeros_like(self.weights2)

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
        
        # Compute the derivative of the output layer and update weights with momentum
        output_delta = output_error * sigmoid_derivative(output)
        update_w2 = learning_rate * np.dot(self.hidden.T, output_delta)
        self.weights2 += update_w2 + self.momentum * self.prev_update_w2
        self.prev_update_w2 = update_w2
        
        # Compute the derivative of the hidden layer and update weights with momentum
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        update_w1 = learning_rate * np.dot(X.T, hidden_delta)
        self.weights1 += update_w1 + self.momentum * self.prev_update_w1
        self.prev_update_w1 = update_w1
        if self.momentum > .5:
            self.momentum *= .9
        
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
data = np.loadtxt('L30fft150.out', skiprows=1)

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
hidden_size = 8
output_size = 1
num_epochs = 25
learning_rate = 0.1
momentum = 0

# Create an instance of the NeuralNetwork class
nn = NeuralNetwork(input_size, hidden_size, output_size, momentum)

# Train the network on the three permutations of groups
error_per_epoch = []
for i in range(2):
    momentum = 0 if i == 0 else 0.9
    for X_train, y_train, X_test, y_test in [(X1, y1, np.concatenate([X2,X3]), np.concatenate([y2,y3])),
                                            (X2, y2, np.concatenate([X1,X3]), np.concatenate([y1,y3])),
                                            (X3, y3, np.concatenate([X1,X2]), np.concatenate([y1,y2]))]:
        nn = NeuralNetwork(input_size, hidden_size, output_size, momentum)
        print(momentum)
        errors = []
        for i in range(num_epochs):
            output = nn.forward(X_train)
            error = np.mean((output - y_train) ** 2)
            nn.backward(X_train, y_train, output, learning_rate)
            errors.append(error)
            if i % (num_epochs / 10) == 0:
                print(f"Epoch {i}, MSE: {error}")
        error_per_epoch.append(errors)


labels = ['AB', 'AC', 'BC', 'AB_mom', 'AC_mom', 'BC_mom']

fig, ax = plt.subplots()

for i in range(len(error_per_epoch)):
    normalized_errors = np.array(error_per_epoch[i]) / np.max(error_per_epoch)
    ax.plot(range(num_epochs), normalized_errors, label=labels[i])

ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Normalized Mean Squared Error')

plt.show()



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
