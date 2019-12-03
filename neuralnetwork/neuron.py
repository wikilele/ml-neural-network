class InputNeuron:

    def __init__(self):

    def compute_output(self, input)
        return input

class Neuron(InputNeuron):

    def __init__(self, bias, weigths, activation_function):
        self.bias = bias
        self. weights =  weights
        self.activation_function = activation_function
        self.output = None
    
    def compute_output(self, input):
        return self.activation_function.compute_output(net_input(input))
    
<<<<<<< HEAD
    def net_input(inputs):
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += self.weights[i]*inputs[i]
=======
    def net_input(input):
        weighted_sum = 0;
        for i in range(len(input)):
            weighted_sum += self.weights[i]*input[i];
>>>>>>> fb562e4b38623fee0f7f127e3dd00fc7da927af9
        return weighted_sum + self.bias

    def dE_dout(self,target_output):
        ''' derivative of error w.r.t. output. The constant 2 is simplified with 1/2 (see notes and self.mean_square_error) '''
        return - (target_output- self.output)

    def dout_dnet(self):
        ''' first derivative of logistic function '''
        return activation_function.first_derivative(self.output)

    def dnet_dwj(self,j):
        ''' just input at position j'''
        return self.inputs[j]

    def dE_dwj(self,target_output,j):
        ''' by chain rule ''' 
        return self.dE_dout(target_output) * self.dout_dnet() * self.dnet_dwj(j)

    def __str__(self):
        return "Bias: " + self.bias + "Weights: " + (str(float(w)) for w in self.weights)
    
    class OutputNeuron(Neuron):

        def __init__(self, bias, weights, activation_function):
            Neuron.__init__(self, bias, weights, activation_function)
            self.back_prop_delta = None
            self.error_delta  = None

        def compute_back_prop_delta(self, target):
            pass

        def compute_error_delta(self, target):
            pass

        def mean_square_error(self, target):
            return 0.5*self.error_delta^2

        def root_mean_square_error(self, target)
            return math.root(self.mean_square_error(self, target))

        def mean_euclidian_error(self, target)
            pass