import math

def get_activation_function(ftype):
    if ftype == 'sigmoid':
        return Sigmoid()
    elif ftype == 'linear':
        return Linear()
    else:
        return ActivationFunction()

class ActivationFunction:

    def compute_output(self, input):
        raise NotImplementedError
    
    def first_derivative(self, input):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def compute_output(self, input):
        res = 1/(1 + math.exp(-input))
        return res
 
    def first_derivative(self, input):
        tmp = self.compute_output(input)
        return tmp * (1 - tmp) 

class Linear(ActivationFunction):
    def compute_output(self, input):
        return input
    
    def first_derivative(self, input): # TODO i think this is correct but better to check
        return 1