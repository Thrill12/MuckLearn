import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, learning_rate, num_inputs, hidden_layers, momentum=0.9, name="default"):
        # Initialize the neural network with the given parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = [num_inputs] + hidden_layers + [1]  # Input layer → Hidden layers → Output layer
        self.name = name.lower()

        self.act_function = self._leaky_relu
        self.act_function_deriv = self._leaky_relu_deriv

        # Randomize the seed
        self.seed = np.random.randint(0, 100000)
        np.random.seed(self.seed)

        # Initialize weights and biases at random
        #self.weights = [np.random.rand(self.layers[i], self.layers[i + 1]) * 0.01 for i in range(len(self.layers) - 1)]
        #self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        # Initialize weights using Xavier initialization
        #self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / (self.layers[i] + self.layers[i + 1]))for i in range(len(self.layers) - 1)]
        #self.biases = [np.zeros((1, self.layers[i + 1]))for i in range(len(self.layers) - 1)]

        # Initializing weights and biases with Kaiming/HE initialization
        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]) for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

        # Weight dimensions and meanings
        # weights[i][j][k] → Weight from node j in layer i to node k in layer i + 1
        #for i in range(len(self.weights)):
        #    for j in range(len(self.weights[i])):
        #        print("Weights for layer ", i, "node ", j, ":", self.weights[i][j])

        # Initialize momentum velocities for weights and biases
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]

        self.last_rmse = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        z = self._sigmoid(x)
        return z * (1 - z)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_deriv(self, x):
        return 1 - np.tanh(x) ** 2
    
    def _leaky_relu(self, x):
        return np.maximum(0.1 * x, x)
    
    def _leaky_relu_deriv(self, x):
        return np.where(x > 0, 1, 0.1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return np.where(x > 0, 1, 0)
    
    def _forward_pass(self, input_vector):
        # Z values are the pre-activation outputs for each layer
        z_values = []
        # Activations are the outputs of each neuron after applying the activation function
        activations = [np.array(input_vector).reshape(1, -1)]  # Start with the input layer

        # w: weight matrix
        # b: bias vector
        # i: index of the current layer
        # This iterates over the weight and bias pairs for each layer
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):

            # Compute the pre-activation output for the current layer
            z = np.dot(activations[-1], w) + b
            z_values.append(z)
            if i < len(self.weights) - 1:  # Apply activation only to hidden layers
                a = self.act_function(z)
            else:
                a = z  # For the output layer (assuming linear activation)
            activations.append(a)
            
        return z_values, activations
    
    def forward_pass_batch(self, input_batch):
        # input_batch is of shape (num_samples, input_dim)
        activations = [input_batch]
        z_values = []
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], weight) + bias  # vectorized computation for the whole batch
            z_values.append(z)
            if i == len(self.weights) - 1:  # For the output layer, use linear activation
                activations.append(z)
            else:
                activation = self.act_function(z)  # apply activation for hidden layers
                activations.append(activation)
        return z_values, activations


    def compute_rmse(self, predictions, targets):
        # Compute the RMSE from these two arrays
        return np.sqrt(np.mean((predictions - targets) ** 2))

    # Backpropagation for a single sample
    def _backpropagate(self, z_values, activations, target):
        # Steps
        # 1. Compute the output layer error
        # 2. Backpropagate the error (delta) through the hidden layers
        # 3. Compute the gradients for weights and biases
        # 4. Compute derivative of current layer
        # 5. Compute delta for the next layer

        # Gradient arrays for weights and biases
        # [Same dimensions as original weights and biases]
        derror_dweights = [np.zeros_like(w) for w in self.weights]
        derror_dbiases = [np.zeros_like(b) for b in self.biases]

        # Compute output layer error
        output = activations[-1]
        error = 2 * (output - target)  # Derivative of MSE loss

        # For a linear output layer, delta is simply the error.
        delta = error  

        # Backpropagate through hidden layers
        for i in reversed(range(len(self.weights))):

            # Compute gradients for weights and biases
            # Gradient is computed by multiplying the delta with the activations of the previous layer
            derror_dweights[i] = np.dot(activations[i].T, delta)

            # Gradient for biases is simply the delta
            # Because each bias affects all outputs in that layer equally
            derror_dbiases[i] = np.sum(delta, axis=0, keepdims=True)

            # Do not apply activation function for the input layer
            if i > 0:
                # Compute the derivative of the activation function for the current layer
                dactivation = self.act_function_deriv(z_values[i-1])
                # Compute the delta for the next layer
                delta = np.dot(delta, self.weights[i].T) * dactivation

        # Return the computed gradients
        return derror_dweights, derror_dbiases

    # Backpropagation for a mini-batch
    def _backpropagate_batch(self, z_values, activations, targets):
        batch_size = targets.shape[0]
        derror_dweights = [np.zeros_like(w) for w in self.weights]
        derror_dbiases = [np.zeros_like(b) for b in self.biases]

        # Compute output error (for linear output layer)
        output = activations[-1] 
        error = 2 * (output - targets)  # derivative of MSE loss
        delta = error

        # Backpropagation through layers (in reverse order)
        for i in reversed(range(len(self.weights))):
            # Compute gradients for weights and biases; averaging over the batch
            derror_dweights[i] = np.dot(activations[i].T, delta) / batch_size
            derror_dbiases[i] = np.sum(delta, axis=0, keepdims=True) / batch_size

            if i > 0:
                # Get derivative of the activation function for the current layer
                dactivation = self.act_function_deriv(z_values[i - 1])
                delta = np.dot(delta, self.weights[i].T) * dactivation

        return derror_dweights, derror_dbiases

    def _update_parameters(self, derror_dweights, derror_dbiases):
        max_grad_norm = 2.0  # Adjust this threshold as needed
        for i in range(len(self.weights)):
            # Clip gradients for weights
            grad_norm = np.linalg.norm(derror_dweights[i])
            if grad_norm > max_grad_norm:
                derror_dweights[i] = derror_dweights[i] * (max_grad_norm / grad_norm)
            
            # Clip gradients for biases (if needed)
            grad_norm_bias = np.linalg.norm(derror_dbiases[i])
            if grad_norm_bias > max_grad_norm:
                derror_dbiases[i] = derror_dbiases[i] * (max_grad_norm / grad_norm_bias)
            
            # Update velocity (momentum) for weights and biases
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * derror_dweights[i]
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * derror_dbiases[i]
            
            # Update weights and biases using the new velocity
            self.weights[i] -= self.velocity_weights[i]
            self.biases[i] -= self.velocity_biases[i]

    def _predict(self, input_vector, output_scaler):
        # Perform a forward pass to get the final output
        z_values, activations = self._forward_pass(input_vector)

        standardised_output = activations[-1].flatten()

        # Inverse transform the output to get the original scale
        output = output_scaler.inverse_transform(standardised_output.reshape(-1, 1)).flatten()

        return output

    def train(self, input_vectors, targets, epochs, input_scaler, output_scaler, mini_batch_size, val_input, val_target):
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        cumulative_errors = []
        validation_rmse = []
        log = []
        start_time = datetime.datetime.now()

        # Inverse transform targets for error computation in original scale
        inverse_transformed_targets = self.output_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

        num_samples = len(input_vectors)

        for epoch in range(epochs):
            # Shuffle the dataset at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            input_vectors_shuffled = input_vectors[indices]
            targets_shuffled = targets[indices]

            # Process mini-batches
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_inputs = input_vectors_shuffled[start:end]
                # reshape the targets to ensure shape is (batch_size, 1)
                batch_targets = targets_shuffled[start:end].reshape(-1, 1) 

                # Forward pass for the mini-batch
                z_values, activations = self.forward_pass_batch(batch_inputs)
                # Backpropagation for the mini-batch
                derror_dweights, derror_dbiases = self._backpropagate_batch(z_values, activations, batch_targets)
                # Update parameters using the computed gradients
                self._update_parameters(derror_dweights, derror_dbiases)

            # Optional: Evaluate error on the entire dataset at the end of each epoch
            z_values_full, activations_full = self.forward_pass_batch(input_vectors)
            predictions = activations_full[-1].flatten()
            # Inverse transform the predictions
            inverse_transformed_predictions = self.output_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()      

            rmse = self.compute_rmse(inverse_transformed_predictions, inverse_transformed_targets)
            cumulative_errors.append(rmse)

            self.last_rmse = rmse

            # Learning rate annealing
            self.learning_rate *= 1 - (1 / epochs)
            if epoch % 100 == 0:

                val_rmse = self.validate(val_input, val_target, self.output_scaler)

                validation_rmse.append(val_rmse)

                log_string = "Epoch: {}, Training RMSE: {:.4f}, Validation RMSE: {:.4f}".format(epoch, rmse, val_rmse)
                log.append(log_string)

        end_time = datetime.datetime.now()
        training_time = end_time - start_time
        print("Training time:", training_time, "\n")
        
        return cumulative_errors, log, training_time, validation_rmse
    
    def validate(self, val_inputs, val_targets, scaler_output):
        # Perform a forward pass to get the final output
        z_values, activations = self._forward_pass(val_inputs)

        standardised_output = activations[-1].flatten()

        # Inverse transform the output to get the original scale
        output = scaler_output.inverse_transform(standardised_output.reshape(-1, 1)).flatten()

        # Compute RMSE for validation set
        rmse = self.compute_rmse(output, val_targets.flatten())

        print("Validation RMSE:", rmse)

        return rmse