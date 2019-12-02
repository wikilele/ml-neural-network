import math

class ActivationFunction:

    def compute_output(self, input):
        raise NotImplementedError
    
    def first_derivative(self, input):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def compute_output(self, input):
        return 1/(1 + math.exp(-input))
    
    def first_derivative(self, input):
        return input * (1 - input)