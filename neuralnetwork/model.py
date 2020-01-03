from .backprop_service import BackPropService
from utils import printProgressBar
class Model():

    def __init__(self, model, learning_rate, tau_decay , momentum_alpha, use_nesterov):
        self.model = model
        self.learning_rate0 = learning_rate
        self.learning_rate_tau = self.learning_rate0/100
        self.tau = tau_decay
        self.outputs = []
        self.backprop_service = BackPropService(model, momentum_alpha, use_nesterov)
        

    def fit(self, training_set, dim_batch ,epoch):

        training_set.shuffle()
        for batch in training_set.batch(dim_batch): 
                
            self.backprop_service.batch_starting()
                
            for pattern in batch:
                self.feed_forward(pattern[1])
                self.backprop_service.compute_deltas(self.model, pattern[2])

            eta = self._update_learning_rate(epoch)
            self.backprop_service.update_weights(self.model, eta, dim_batch)
                
            self.backprop_service.batch_ending()
    


    def classify(self, input, threshold = 0):
        out = self.feed_forward(input)
        if out[0] >= threshold:
            return 1
        else:
            return 0
    
    def feed_forward(self, input):
        temp_input = input
        for layer in self.model:
            temp_input = layer.feed_forward(temp_input)
        
        return temp_input
    
    def forward_dataset(self,ds):
        outputs = []
        for batch in ds.batch(1):
            for pattern in batch:
                outputs.append(self.feed_forward(pattern[1])) 
        return outputs
    
    def _update_learning_rate(self, epoch):
        if self.tau == 0:
            # no learning rate decay
            return self.learning_rate0
        elif epoch >= self.tau:
            return self.learning_rate_tau
        else:
            return (1 - epoch/self.tau)*self.learning_rate0 + (epoch/self.tau) * self.learning_rate_tau
