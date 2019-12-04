import numpy as np

class BackPropService:

    def __init__(self, model, deltas):

        self.deltas = []
        prev_len = len(model[0])
        for layer in model[1:]:
            self.deltas.append(np.zeros(len(layer.neurons), prev_len ))
            prev_len = len(layer.neurons)

    def update_weights(self, model):

    def compute_deltas(self, model, output):
        prev_deltas = np.array(len(output))
        output_layer = model[-1]
        rev_hidden_layers = reversed(model[1:])
        weights = []

        #deltas for output layer
        for i in range(output_layer):
            prev_deltas[i] = output_layer[i].compute_back_prop_delta(output[i])
        
        for i in range(len(prev_deltas)):
            for j in range(np.size(self.deltas[-1], 0)): #check dimensions
                self.deltas[-1][i][j] += prev_deltas[i]*output_layer[i].dnet_dwj(j)

        #deltas for hidden layers
        iterator = enumerate(rev_hidden_layers)
        next(iterator)
        for layer_index, layer in iterator:
            for neuron_index in range(len(layer)):
                for prev_neuron in rev_hidden_layers[layer_index-1]:
                    weights.append(prev_neuron.weights[neuron_index])

                prev_deltas[neuron_index] = neuron.compute_back_prop_delta(prev_deltas, weights)

            for i in range(len(prev_deltas)):
                for j in range(np.size(self.deltas[-layer_index], 0)): #check dimensions
                    self.deltas[-layer_index][i][j] += prev_deltas[i]*rev_hidden_layers[layer_index][i].dnet_dwj(j)


