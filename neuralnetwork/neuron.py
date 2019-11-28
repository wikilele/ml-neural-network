class Neuron:

    def __init__(self, bias, weigths, activation_function):
        self.bias = bias
        self. weights =  weights
        self.activation_function = activation_function
    
    def compute_output(inputs):
        return self.activation_function(net_input(inputs))
    
    def net_input(inputs):
        weighted_sum = 0;
        for i in range(len(inputs)):
            weighted_sum += self.weights[i]*inputs[i];
        return weighted_sum + self.bias
    
    def __str__(self):
        return "Bias: " + self.bias + "Weights: " + (str(float(w)) for w in self.weights)