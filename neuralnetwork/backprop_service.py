import numpy as np

class BackPropService:

    def __init__(self, model,momentum_alpha, use_nesterov):
        self.momentum_alpha = momentum_alpha
        self.use_nesterov = use_nesterov
        # this matrix stores the DELTAS used to update the weights
        # do not confuse DELTA with delta
        # DELTA are defined as delta * input from previous node
        # delta is used to compute the DELTA and changes between hidden and output neurons
        self.DELTAS = []
        prev_len = len(model[0].neurons)
        for layer in model[1:]:
            # prev_len + 1 beacuse we consider also the bias
            self.DELTAS.append(np.zeros( (len(layer.neurons), prev_len + 1) ))
            prev_len = len(layer.neurons)
        
        if not use_nesterov:
            # the DELTAS OLD matrix will be used for standard momentum
            self.DELTAS_OLD = self.DELTAS.copy()
        
    def update_weights(self, model, learning_rate):
        
        for layer_index, layer in enumerate(model[1:]):
            for neuron_index, neuron in enumerate(layer.neurons):
                for w_index, w in enumerate(neuron.weights):
                    w += learning_rate * self.DELTAS[layer_index][neuron_index][w_index] # regularization

                    if not self.use_nesterov:
                        # updating with standard momentum
                        w += self.momentum_alpha * self.DELTAS_OLD[layer_index][neuron_index][w_index]
                    
                    # w += regularization
        
        if not self.use_nesterov:
            self.DELTAS_OLD = self.DELTAS.copy()
    
    def _update_DELTAS_matrix(self,deltas,layer_index, layer):
        # for each computed delta (one delta per neuron)
        for i in range(len(deltas)):
            # for each weight j of the neuron i. 
            for j in range(np.size(self.DELTAS[layer_index], 1)):
                # update the DELTA of weight j from neuron i of layer     
                self.DELTAS[-1][i][j] += deltas[i] * layer.neurons[i].dnet_dwj(j)

    def compute_deltas(self, model, target_output):
        '''
        this function will be called each time the model is feeded with the input.
        the function will compute the DELTAS for the back propagation.
        first the output deltas will be computed and then backwards the hidden deltas
        from the last hidden layer to the first one
        '''
        # we need something to keep the deltas of the previous layer
        prev_deltas = np.array([])
        output_layer = model[-1]
        # this variable keep the model reverted so that we can iterate on that for the backprop
        # the output layer is considered, the input layer is removed cause it's not needed
        rev_hidden_layers = reversed(model[1:])
        weights = []

        # deltas for output layer
        for i in range(len(output_layer.neurons)):
            # for each neuron in the output layer we get it's delta
            o_delta =  output_layer.neurons[i].compute_back_prop_delta(target_output[i])
            prev_deltas = np.append(prev_deltas, o_delta)
            
        
        self._update_DELTAS_matrix(prev_deltas, -1, output_layer)

        #deltas for hidden layers
        iterator = enumerate(rev_hidden_layers)
        # we skip the output layer 
        next(iterator)
        # for each hidden layer
        for layer_index, layer in iterator:
            # to keep the deltas in this iteration
            current_deltas = np.array([])
            # for each neuron in the hidden layer
            for neuron_index in range(len(layer.neurons)):
                # for each neuron in the previous layer (remember the model here is reversed so output layer comes before hidden)
                for prev_neuron in rev_hidden_layers[layer_index-1].neurons:
                    # we get the weight between our current neuron (neuron index)
                    # and the neuron in the previous layer
                    weights.append(prev_neuron.weights[neuron_index])

                # get the delta from the hidden neuron
                h_delta = neuron.compute_back_prop_delta(prev_deltas, weights)
                current_deltas = np.append(current_deltas, h_delta)
            
            self._update_DELTAS_matrix(current_deltas, -layer_index, layer)
            
            # in the next iteration we will use the deltas of this iteration
            prev_deltas = current_deltas
            
    def batch_starting(self):   
        if self.use_nesterov:
            for layer_index, layer in enumerate(model[1:]):
                for neuron_index, neuron in enumerate(layer.neurons):
                    for w_index, w in enumerate(neuron.weights):
                        w += self.momentum_alpha * self.DELTAS[layer_index][neuron_index][w_index]

    def batch_ending(self):
        for matrix in self.DELTAS:
            matrix.fill(0)