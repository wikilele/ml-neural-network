class Metrics:
    def __init__(self):
        self.acc = 0
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
        self.acc = correct_outputs/len(output) * 100
        return self.acc

    def precision(self,output, target_output):
        pass
    
    def recall(self, output, target_output):
        pass
    
    def mean_square_error(self, output, target_output):
        mse = 0
        temp = 0
        for i in range(len(output)):
            for j in range(len(output[i])):
                temp += (target_output[i][j] - output[i][j]) **2

        self.mse.append(temp/len(output))
        return self.mse[-1]
    
    def mean_euclidian_error(self, output, target_output):
        pass
    
    def root_mean_square_error(self, output, target_output):
        pass

    def save(self, file_path):
        header = "mean_square_error"
        return