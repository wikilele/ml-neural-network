from .model import Model
from .layer import *
from .weights_service import WeightsService
from .activation_function import get_activation_function

class ModelBuilder():

    def __init__(self):
        # init weight service with default parameters
        self.ws = WeightsService(-0.7,0.7) 
        self.model = []


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
        return Model(self.model)
  
    def reset(self):
        self.model = []
    
    def weights_service(self,ws):
        self.ws = ws
        return self