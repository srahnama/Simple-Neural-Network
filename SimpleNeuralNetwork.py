import numpy as np

class NeuralNetwork:

    def __init__(self, *args, **kwargs):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1
        
    def train(self, inputs, outputs, num):

        for iteration in range(num):

            output = self.think(outputs)
            error = outputs - output
            adjustment = np.dot(inputs.T, error * self.__sigmond_derivative(output))
            self.weights += adjustment

    def think(self, inputs):
        return self.__sigmond(np.dot(inputs.T , self.weights))

    def __sigmond_derivative(self, x):
        return x * (1 - x)

    def __sigmond(self, x):
        return 1 /(1 + np.exp(-x))
    
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.weights)

    training_inputs = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[1, 1, 0]]).T
    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.weights)

    A = float(input("Input 1: "))
    B = float(input("Input 2: "))
    C = float(input("Input 3: "))

    print("New situation -> input data = [", A, B, C,"]")


    print("Output: ", neural_network.think(np.array([A, B, C]))[0])