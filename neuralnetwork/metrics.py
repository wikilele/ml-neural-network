class Metrics:
    def accuracy(self, output, target_output):
        correct_outputs = 0
        for i in range(output):
            if (output[i] == target_output[i]):
                correct_outputs += 1
        
        return correct_outputs/len(output)

    def precision(self, output, target_output):
        pass
    
    def recall(self, output, target_output):
        pass
    
    def mean_square_error(self, output, target_output):
        pass
    
    def mean_euclidian_error(self, output, target_output):
        pass
    
    def root_mean_square_error(self, output, target_output):
        pass