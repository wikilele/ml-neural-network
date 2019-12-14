class Metrics:
    accuracy = None
    precision = None
    recall = None

    mse = None
    mee = None
    rmse = None


    @classmethod
    def accuracy(self, output, target_output):
        correct_outputs = 0
        for i in range(output):
            if (output[i] == target_output[i]):
                correct_outputs += 1
        
        return correct_outputs/len(output)

    @classmethod
    def precision(self,output, target_output):
        pass
    
    @classmethod
    def recall(self, output, target_output):
        pass
    
    @classmethod
    def mean_square_error(self, output, target_output):
        pass
    
    @classmethod
    def mean_euclidian_error(self, output, target_output):
        pass
    
    @classmethod
    def root_mean_square_error(self, output, target_output):
        pass

    @classmethod
    def save(self, file_path):
        header = "mean_square_error"
        return