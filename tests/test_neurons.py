import unittest
from neuralnetwork.neuron import *
from neuralnetwork.activation_function import get_activation_function



class TestNeurons(unittest.TestCase):
    
    def test_output_neuron_backprop_delta(self):
        af = get_activation_function("linear")
        on = OutputNeuron(0,[1],af)
        on.compute_output([42])

        assert on.compute_back_prop_delta(45) == 3