from .Model import Model

class ModelBuilder():

    def __init__(self):
        pass

    def input_layer(self,units):
        print("input layer" + str(units))
        return self
    
    def hidden_layer(self,units,activation='sigmoid'):
        print("hidden layer" + str(units) + " activation " + activation) 
        return self
    
    def output_layer(self, units, activation='linear'):
        print("output layer" + str(units) + " activation " + activation)
        return self
    
    def build(self):
        return Model()
    
    def reset(self):
        pass