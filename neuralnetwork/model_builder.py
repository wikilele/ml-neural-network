from .model import Model
from .layer import *
from .weights_service import WeightsService
from .activation_function import get_activation_function

class ModelBuilder():

    def __init__(self):
        self.model = []
        # init hyperparams with default value
        self.ws = WeightsService(-0.7,0.7)
        self.learning_r = 0.7
        self.tau_decay = 0 # value for learning rate decay 
        self.momentum_alpha = 0 # no momentum
        self.use_nesterov = False
        self.regularization_labda = 0

        
    def input_layer(self,units):
        # print("input layer" + str(units))
        input_layer = InputLayer(units)
        self.model.append(input_layer)

        return self
    
    def hidden_layer(self,units,activation='sigmoid'):
        # print("hidden layer" + str(units) + " activation " + activation) 

        # each neuron will have as much weights as the lenght of the previous layer
        prev_layer_len = len(self.model[-1].neurons)
        activation_function = get_activation_function(activation)
        hidden_layer = Layer(activation_function,units,prev_layer_len, self.ws)
        
        self.model.append(hidden_layer)
        return self
    
    def output_layer(self, units, activation='linear'):
        # print("output layer" + str(units) + " activation " + activation)
        # each neuron will have as much weights as the lenght of the previous layer
        prev_layer_len = len(self.model[-1].neurons)
        activation_function = get_activation_function(activation)
        output_layer = OutputLayer(activation_function,units,prev_layer_len, self.ws)
        
        self.model.append(output_layer)
        return self
    
    def build(self):
        return Model(self.model, self.learning_r, self.tau_decay, self.momentum_alpha, self.use_nesterov, self.regularization_labda)

    
    def init_weights_random(self, bound):
        #TODO discuss if ws really needed or just an overkill
        self.ws = WeightsService(-bound, bound)
        return self
    
    def learning_rate(self, lr, tau_decay):
        self.learning_r = lr
        self.tau_decay = tau_decay
        return self
    
    def momentum(self, alpha, use_nesterov):
        self.momentum_alpha = alpha
        self.use_nesterov = use_nesterov
        return self
