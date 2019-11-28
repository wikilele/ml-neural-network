class Layer:

    def __init__(activation_function, neurons_number, weights_number, WeightsService):
        self.activation_function = activation_function
        self.output = []
        self.neurons = []
        for i in range(neurons_number):
            self.neurons.append(Neuron(Weights_Service.get_bias(), Weights_Service.get_weights(weights_number), activation_function))
    
    def feed_forward(input):
        self.output = []
        for neuron in self.neurons:
            self.output.append(neuron.compute_output(input))
        return self.output