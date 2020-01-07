from math import sqrt

class Metrics:
    def __init__(self):
        self.acc = []
        self.prec = 0
        self.rec = 0

        self.mse = []
        self.mee =  []
        self.rmse = []

    def accuracy(self, output, target_output):
        correct_outputs = 0
        for i in range(len(output)):
            if (output[i] == target_output[i]):
                correct_outputs += 1
        tmpacc = correct_outputs/len(output) * 100
        self.acc.append(tmpacc)
        return self.acc[-1]

    def precision(self,output, target_output):
        true_positive = 0
        false_positive = 0
        for i in range(len(output)):
            if(output[i] == 1):
                if(target_output[i] == 1):
                    true_positive += 1
                else:
                    false_positive += 1
        
        self.prec = true_positive/(true_positive+false_positive)
        return self.prec
    
    def recall(self, output, target_output):
        true_positive = 0
        false_negative = 0
        for i in range(len(output)):
            if(output[i] == target_output[i] and output[i] == 1):
                true_positive += 1
            elif(output[i] != target_output[i] and output[i] == 0):
                false_negative += 1
        
        self.rec = true_positive/(true_positive+false_negative)
        return self.rec
    
    def mean_square_error(self, output, target_output):
        mse = 0
        temp = 0
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp += (target_output[i][j] - output[i][j]) **2

        self.mse.append(temp/len(output))
        return self.mse[-1]
    
    def mean_euclidian_error(self, output, target_output):
        mee = 0
        tmp = 0
        for i in range(len(output)):
            for j in range(len(output[i])):
                tmp += sqrt((target_output[i][j] - output[i][j])**2)

        self.mee.append(tmp/len(output))
        return self.mee[-1]
    
    def root_mean_square_error(self, output, target_output):
        pass

    def compute_error(self, outputs, dataset, metrics=['mse']):            
        target_outputs = dataset.data_set[:,2]
        for metric in metrics:
            if metric == 'mse':
                self.mean_square_error(outputs, target_outputs)
            elif metric == 'mee':
                self.mean_euclidian_error(outputs, target_outputs)
            elif metric == 'rmse':
                self.root_mean_square_error(outputs, target_outputs)
        
        return self

    #TODO this function name is orrible, I know it, need to find a better name to refer to acc, prec, rec
    def compute_other(self, outputs, dataset, metrics=['acc'], threshold=0):
        classification_outputs = []
        for o in outputs:
            if o[0] >= threshold:
                classification_outputs.append(1)
            else:
                classification_outputs.append(0)

        target = [ x[0] for x in dataset.data_set[:,2]]
        for metric in metrics:
            if metric == 'acc':
                self.accuracy(classification_outputs, target)
            elif metric == 'prec':
                self.precision(classification_outputs, target)
            elif metric == 'rec':
                self.recall(classification_outputs, target)
        
        return self