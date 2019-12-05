from .neuron import *

class InputLayer:
    def __init__(self,neurons_number):
        self.neurons = []
        for i in range(neurons_number):
            self.neurons.append(InputNeuron())
    
    def feed_forward(self,input):
        output = []
        for neuron in self.neurons:
            output.append(neuron.compute_output(input))
        return output

class Layer(InputLayer):

    def __init__(self, activation_function, neurons_number, weights_number, weights_service):
        self.activation_function = activation_function
        self.neurons = []
        for i in range(neurons_number):
            self.neurons.append(Neuron(weights_service.get_bias(), weights_service.get_weights(weights_number), activation_function))


class OutputLayer(Layer):

    def __init__(self, activation_function, neurons_number, weights_number, weights_service):
        self.activation_function = activation_function
        self.neurons = []
        for i in range(neurons_number):
            self.neurons.append(OutputNeuron(weights_service.get_bias(), weights_service.get_weights(weights_number), activation_function))
    
    