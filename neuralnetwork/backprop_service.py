import numpy as np

class BackPropService:

    def __init__(self, model, deltas):

        self.deltas = []
        prev_len = len(model[0])
        for layer in model[1:]:
            self.deltas.append(np.matrix(len(layer.neurons), prev_len ))
            prev_len = len(layer.neurons)

    def update_weights(self, model):

    def compute_deltas(self, model):
