from .backprop_service import BackPropService
from .metrics import Metrics 
class Model():

    def __init__(self, model, learning_rate, tau_decay , momentum_alpha, use_nesterov):
        self.model = model
        self.learning_rate0 = learning_rate
        self.learning_rate_tau = self.learning_rate0/100
        self.tau = tau_decay
        self.outputs = []
        self.backprop_service = BackPropService(model, momentum_alpha, use_nesterov)
        

    def fit(self, training_set, dim_batch ,epochs, metrics=['mse']):
        metrics_o = Metrics()

        for e in range(epochs):
            printProgressBar(e + 1, epochs, prefix = 'Fitting:', suffix = 'Complete')
            
            self.outputs = []
            training_set.shuffle()
            for batch in training_set.batch(dim_batch): 
                
                self.backprop_service.batch_starting()
                
                for pattern in batch:
                    self.feed_forward(pattern[1])
                    self.backprop_service.compute_deltas(self.model, pattern[2])

                eta = self._update_learning_rate(e)
                self.backprop_service.update_weights(self.model, eta, dim_batch)
                
                self.backprop_service.batch_ending()

            for batch in training_set.batch(1):
                for pattern in batch:
                    self.outputs.append(self.feed_forward(pattern[1]))
            
            target_outputs = training_set.data_set[:,2]
            for metric in metrics:
                if metric == 'mse':
                    metrics_o.mean_square_error(self.outputs, target_outputs)
                elif metric == 'mee':
                    metrics_o.mean_euclidian_error(self.outputs, target_outputs)
                elif metric == 'rmse':
                    metrics_o.root_mean_square_error(self.outputs, target_outputs)
                
        return metrics_o

    def evaluate(self,data_iterator):
        #Compute accuracy, precision and recall
        pass
    
    def feed_forward(self, input):
        temp_input = input
        for layer in self.model:
            temp_input = layer.feed_forward(temp_input)
        
        return temp_input
    
    def _update_learning_rate(self, epoch):
        if self.tau == 0:
            # no learning rate decay
            return self.learning_rate0
        elif epoch >= self.tau:
            return self.learning_rate_tau
        else:
            return (1 - epoch/self.tau)*self.learning_rate0 + (epoch/self.tau) * self.learning_rate_tau









def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()