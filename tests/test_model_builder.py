import unittest
from neuralnetwork.model_builder import ModelBuilder
from neuralnetwork.neuron import *
from neuralnetwork.activation_function import *

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.mb = ModelBuilder()
    
    def test_input_layer(self):

        self.mb.input_layer(3)

        assert len(self.mb.model[0].neurons) == 3
        assert type(self.mb.model[0].neurons[0]) == InputNeuron
    
    def test_hidden_layer(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5)

        assert len(self.mb.model[-1].neurons) == 5
        assert type(self.mb.model[-1].neurons[0]) == Neuron
    
    def test_output_layer(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5)
        self.mb.output_layer(2)

        assert len(self.mb.model[-1].neurons) == 2
        assert type(self.mb.model[-1].neurons[0]) == OutputNeuron
    
    def test_activation_function_init(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5,activation='linear')     
        assert type(self.mb.model[-1].neurons[0].activation_function) == Linear

        self.mb.output_layer(2,activation='sigmoid')
        assert type(self.mb.model[-1].neurons[0].activation_function) == Sigmoid
    
    def test_weights_init(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5)      
        assert len(self.mb.model[-1].neurons[0].weights) == 4 # bias included

        self.mb.output_layer(2)
        assert len(self.mb.model[-1].neurons[0].weights) == 6
    
    def test_model1(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(3)
        self.mb.hidden_layer(3)
        self.mb.output_layer(2)

        assert len(self.mb.model[-1].neurons) == 2
        assert len(self.mb.model[-2].neurons) == 3
        assert len(self.mb.model[-3].neurons) == 3
        