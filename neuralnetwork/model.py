class Model():

    def __init__(self, model):
        self.model = model

    def fit(self,data_iterator,epochs):
        for i in range(epochs):
            for batch in data_iterator:
                for pattern in batch:
                    self.feed_forward(pattern)
                    backprop_service.compute_deltas(self.model)

                self.model = backprop_service.update_weights(self.model)

    def infer(self,newdata):
        pass

    def evaluate(self,data_iterator):
        pass
    
    def feed_forward(self, input):
        temp_input = input
        for layer in model:
            temp_input = layer.feed_forward(temp_input)
        
        return temp_input
    
    def iterator(self):
        for layer in self.model:
            yield layer