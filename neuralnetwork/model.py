from .backprop_service import BackPropService

class Model():

    def __init__(self, model, use_nesterov=False):
        self.model = model
        self.backprop_service = BackPropService(model,use_nesterov)
        #TODO pass this to init
        self.learning_rate0 = 0.7
        self.learning_rate_tau = self.learning_rate0/100
        self.tau = 200
        

    def fit(self,data_iterator,epochs):
        for e in range(epochs):
            for batch in data_iterator: 
                
                self.backprop_service.batch_starting()
                
                for pattern in batch:
                    self.feed_forward(pattern)
                    self.backprop_service.compute_deltas(self.model, target_output)

                eta = self._update_learning_rate(e)
                self.backprop_service.update_weights(self.model, eta)
                
                self.backprop_service.batch_ending()
            
            # compute metrics

    def infer(self,newdata):
        pass

    def evaluate(self,data_iterator):
        pass
    
    def feed_forward(self, input):
        temp_input = input
        for layer in self.model:
            temp_input = layer.feed_forward(temp_input)
        
        return temp_input
    
    def _update_learning_rate(self, epoch):
        if epoch >= self.tau:
            return self.learning_rate_tau
        else:
            return (1 - epoch/self.tau)*self.learning_rate0 + (epoch/self.tau) * self.learning_rate_tau