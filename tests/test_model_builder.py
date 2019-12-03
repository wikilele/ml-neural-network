import unittest
from neuralnetwork.model_builder import ModelBuilder

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.mb = ModelBuilder()
    
    def test_input_layer(self):

        self.mb.input_layer(3)

        assert len(self.mb.model[0]) == 3
    
    def test_hidden_layer(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5)

        assert len(self.mb.model[-1]) == 5
    
    def test_output_layer(self):

        self.mb.input_layer(3)
        self.mb.hidden_layer(5)
        self.mb.output_layer(2)

        assert len(self.mb.model[-1]) == 2
