import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Activation function: ReLU and softmax
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function: Cross-entropy
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    logp = - np.log(y_pred[range(n_samples), y_true])
    loss = np.sum(logp) / n_samples
    return loss

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Feedforward Neural Network class
class FeedForwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights using Xavier initialization
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = softmax(self.final_input)
        
        return self.final_output

    def backward(self, X, y_true, output, learning_rate):
        n_samples = X.shape[0]
        
        # Compute gradients
        output_error = output
        output_error[range(n_samples), y_true] -= 1
        output_error /= n_samples
        
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)
        
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * relu_derivative(self.hidden_input)
        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.bias_output -= learning_rate * d_bias_output
        
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden

    def train(self, X, y, X_test, y_test, learning_rate, epochs, batch_size=64, learning_rate_decay=0.99):
        # Set up the plot for real-time update
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))  # Three subplots for loss, train accuracy, and test accuracy
        
        loss_values = []
        train_accuracy_values = []
        test_accuracy_values = []

        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Loss calculation
                loss = cross_entropy_loss(y_batch, output)

                # Backpropagation
                self.backward(X_batch, y_batch, output, learning_rate)
            
            # Update learning rate
            learning_rate *= learning_rate_decay
            
            # Store and plot loss and accuracy every 100 epochs
            if epoch % 100 == 0:
                # Training accuracy
                predictions = np.argmax(output, axis=1)
                train_acc = accuracy(y_batch, predictions)
                
                # Test accuracy
                test_output = self.forward(X_test)
                test_predictions = np.argmax(test_output, axis=1)
                test_acc = accuracy(y_test, test_predictions)

                loss_values.append(loss)
                train_accuracy_values.append(train_acc)
                test_accuracy_values.append(test_acc)

                # Update plots
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                ax1.plot(loss_values, label="Loss")
                ax2.plot(train_accuracy_values, label="Training Accuracy")
                ax3.plot(test_accuracy_values, label="Test Accuracy")
                
                ax1.set_title(f"Epoch {epoch}, Loss: {loss:.4f}")
                ax2.set_title(f"Epoch {epoch}, Training Accuracy: {train_acc:.4f}")
                ax3.set_title(f"Epoch {epoch}, Test Accuracy: {test_acc:.4f}")
                
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Loss")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Training Accuracy")
                ax3.set_xlabel("Epochs")
                ax3.set_ylabel("Test Accuracy")

                ax1.legend()
                ax2.legend()
                ax3.legend()

                plt.pause(0.1)  # Pause to update the plot

        plt.ioff()  # Turn off interactive mode
        plt.show()  # Final plot

# Loading and preprocessing the MNIST dataset
def load_mnist():
    # Load MNIST dataset from TensorFlow (train/test split)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize input data (images) to range [0, 1]
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten the images to 1D (28*28 = 784)
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0     # Flatten test images as well
    
    return X_train, y_train, X_test, y_test

# Example usage:
if __name__ == "__main__":
    # Load the MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist()

    # Create and train the network
    nn = FeedForwardNN(input_size=784, hidden_size=128, output_size=10)
    nn.train(X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=1000, batch_size=128, learning_rate_decay=0.995)

    # Test the network
    output = nn.forward(X_test)
    predictions = np.argmax(output, axis=1)
    acc = accuracy(y_test, predictions)
    print(f"Test accuracy: {acc:.4f}")
