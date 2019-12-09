from .backprop_service import BackPropService

class Model():

    def __init__(self, model, use_nesterov=False):
        self.model = model
        self.backprop_service = BackPropService(model,use_nesterov)
        #TODO pass this to init
        self.learning_rate0 = 0.7
        self.learning_rate_tau = self.learning_rate0/100
        self.tau = 200
        self.outputs = []
        f = open("test.txt", "w+")              #choose filename
        

    def fit(self, training_set, dim_batch ,epochs, metrics=['mse']):
        for e in range(epochs):
            self.outputs = []
            for batch in training_set.batch(dim_batch): 
                
                self.backprop_service.batch_starting()
                
                for pattern in batch:
                    self.feed_forward(pattern)
                    self.backprop_service.compute_deltas(self.model, target_output)

                eta = self._update_learning_rate(e)
                self.backprop_service.update_weights(self.model, eta)
                
                self.backprop_service.batch_ending()
            
            for pattern in training_set.batch(1):
                self.outputs.append(self.feed_forward(pattern))
            
            for metric in metrics:
                if(metric == 'mse'):
                    csv_row += (str(metrics.mean_square_error(self.outputs, training_set[:,2)) + ','
                elif(metric == 'mee'):
                    csv_row += str(metrics.mean_euclidian_error(self.outputs, training_set[:,2])) + ','
                elif(metric == 'rmse'):
                    csv_row += str(metrics.root_mean_square_error(self.outputs, training_set[:,2])) + ','
                
            f.write(csv_row[:-1] +'\n' )

    def evaluate(self,data_iterator):
        #Compute accuracy, precision and recall
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