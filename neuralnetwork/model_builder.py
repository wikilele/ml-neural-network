from .model import Model
from .neuron import *
from .weights_service import WeightsService
from .activation_function import *

class ModelBuilder():

    def __init__(self):
        self.ws = WeightsService(-1,1) #TODO should fix this
        self.model = []


    def input_layer(self,units):
        print("input layer" + str(units))
        input_layer = []
        for i in range(units):
            input_layer.append(InputNeuron())
        
        self.model.append(input_layer)
        return self
    
    def hidden_layer(self,units,activation='sigmoid'):
        print("hidden layer" + str(units) + " activation " + activation) 
        hidden_layer = []
        for i in range(units):
            bias = self.ws.get_bias()
            weights = self.ws.get_weights(len(self.model[-1]))
            activation_function = Sigmoid()
            hidden_layer.append( Neuron(bias,weights,activation_function ))
        
        self.model.append(hidden_layer)
        return self
    
    def output_layer(self, units, activation='linear'):
        print("output layer" + str(units) + " activation " + activation)
        output_layer = []
        for i in range(units):
            bias = self.ws.get_bias()
            weights = self.ws.get_weights(len(self.model[-1]))
            activation_function = Sigmoid()
            output_layer.append( Neuron(bias,weights,activation_function ))
        
        self.model.append(output_layer)
        return self
    
    def build(self):
        return Model(self.model)
    
    def reset(self):
        self.model = []