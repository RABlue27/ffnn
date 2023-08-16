# Multilayer Feed Forward Neural Network Project

Welcome to the Multilayer Feed Forward Neural Network Project! In this project, we have successfully implemented a robust multilayer feed forward neural network to classify electric motors as either "good" or "bad" based on their acceleration data. Here's a brief overview of the features implemented in each part:

## Objective

Our main objective was to design and implement a multilayer feed forward neural network for motor classification. The network architecture consists of three layers: an input layer, a hidden layer, and an output layer. The number of input nodes varies depending on the chosen data file, and the output layer consists of two neurons to differentiate between good and bad motors.

## Data

We worked with acceleration time series data from electric motors. The data was transformed using Fast Fourier Transform (FFT) into frequency spectrum data, generating energy values for different frequency bins. This preprocessing step allowed us to create a meaningful representation of the motor's behavior for classification.

## Architecture

We successfully implemented a 3-layer feed forward neural network with variable input nodes. The network was trained using Backpropagation (BP) for learning. The architecture's flexibility allowed us to adapt to different data file resolutions and effectively capture classification patterns.

## Parts

### Part A: Architecture Design and Training

We discussed and implemented the chosen neural network architecture. The learning rate (alpha) was set for training, and we ensured the network's convergence. This part helped establish a solid foundation for subsequent stages.

### Part B: Proportional Training and Testing

We split the data into training and testing sets, maintaining proportional representation of good and bad motors. By training on two sets and testing on one, we assessed the network's performance. We varied the number of nodes in the hidden layer to explore its impact on generalization.

### Part C: Momentum-Based Training

### Part D: Optimal Training Set Identification

We identified the best training set or combination from Part B or C. By tabulating results for various combinations, we highlighted the optimal configuration for classifying the test set accurately.

### Part E: Network Extension and Evaluation

In this advanced part, we extended the neural network to 5 layers. We evaluated the network's performance using the best and worst combinations from Part D. Classification results were presented using tables, showcasing the network's ability to handle increased complexity.


To improve training time, we implemented momentum-based training. This part involved repeating Part B using the momentum approach. We plotted graphs to visualize global error over epochs and observed how momentum enhanced training efficiency.
