class InputNeuron:

    def __init__(self):
        pass

    def compute_output(self, input):
        return input

class Neuron(InputNeuron):

    def __init__(self, bias, weights, activation_function):
        # self.bias = bias
        self.weights =  [bias] + weights 
        
        self.activation_function = activation_function
        self.inputs = None
        self.output = None
    
    def compute_output(self, input):
        self.inputs = [1] + input
        self.output = self.activation_function.compute_output(self.net_input(self.inputs))
        return self.output
    
    def net_input(self,inputs):
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += self.weights[i]*inputs[i]
        return weighted_sum

    def dE_dout(self,target_output):
        ''' derivative of error w.r.t. output. The constant 2 is simplified with 1/2 (see notes and self.mean_square_error) '''
        return (target_output- self.output) #TODO check the slides, probably the -1 is not required

    def dout_dnet(self):
        ''' first derivative of the activation function '''
        return self.activation_function.first_derivative(self.output)

    def dnet_dwj(self,j):
        ''' just input at position j'''
        return self.inputs[j] 
    
    def compute_back_prop_delta(self, prev_deltas, weights):
        sum = 0
        for i in range(len(prev_deltas)):
            sum += prev_deltas[i]*weights[i]
        return sum*self.dout_dnet()

    # def dE_dwj(self,target_output,j):
    #   ''' by chain rule ''' 
    #    return self.dE_dout(target_output) * self.dout_dnet() * self.dnet_dwj(j)
    
    def __str__(self):
        return "Bias: " + self.bias + "Weights: " + (str(float(w)) for w in self.weights)
    
class OutputNeuron(Neuron):

    def __init__(self, bias, weights, activation_function):
        Neuron.__init__(self, bias, weights, activation_function)
        self.back_prop_delta = None
        self.error_delta  = None

    def compute_back_prop_delta(self, target):
        return self.dE_dout(target)*self.dout_dnet()

    def compute_error_delta(self, target):
        pass

    def mean_square_error(self, target):
        return 0.5*self.error_delta(target)^2

    def root_mean_square_error(self, target):
        return math.root(self.mean_square_error(self, target))

    def mean_euclidian_error(self, target):
        pass