import unittest
from neuralnetwork.model_builder import ModelBuilder
from neuralnetwork.model import Model


class MockWeightsService:

    def get_bias(self):
        return 7
    
    def get_weights(self,weights_number):
        return [w for w in range(1,weights_number+1)]


class TestModel(unittest.TestCase):
    def setUp(self):
        self.ms = ModelBuilder()
    
    def test_feed_forward1(self):
        self.ms.weights_service(MockWeightsService())
        self.ms.input_layer(1)
        self.ms.output_layer(1, activation="linear")
        model = self.ms.build()

        output = model.feed_forward([42])

        assert output[0] == 49