import unittest
from neuralnetwork.model_builder import ModelBuilder
from neuralnetwork.backprop_service import BackPropService
import numpy as np


class MockWeightsService:

    def get_bias(self):
        return 0
    
    def get_weights(self,weights_number):
        return [w for w in range(1,weights_number+1)]


class TestBackPropService(unittest.TestCase):
    def setUp(self):
        self.ms = ModelBuilder()
    

    def test_init(self):
        input_units = 3
        hidden_units = 2
        output_units = 1
        self.ms.input_layer(input_units)
        self.ms.hidden_layer(2)
        self.ms.output_layer(1, activation="linear")
        m = self.ms.build()
        bs = BackPropService(m.model)
        bs_matrix = bs.DELTAS

        assert len(bs_matrix) == 2
        assert np.size(bs_matrix[0],0) == hidden_units
        assert np.size(bs_matrix[0],1) == input_units
        assert np.size(bs_matrix[1],0) == output_units
        assert np.size(bs_matrix[1],1) == hidden_units


    def test_one_layer(self):
        self.ms.weights_service(MockWeightsService())
        self.ms.input_layer(1)
        self.ms.output_layer(1, activation="linear")
        m = self.ms.build()
        bs = BackPropService(m.model)

        m.feed_forward([42])
        bs.compute_deltas(m.model,[45])

        assert bs.DELTAS[0][0][0] == - (45 - 42) * 42
