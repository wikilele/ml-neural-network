import unittest
from neuralnetwork.model_builder import ModelBuilder

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        self.mb = ModelBuilder()
    
    def test_input_layer(self):

        self.mb.input_layer(3)

        assert len(self.mb.model[0]) == 3
