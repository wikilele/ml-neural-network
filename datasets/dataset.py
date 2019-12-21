import random
import numpy as np
class Dataset():

    def __init__(self, numpy_dataset):
        self.data_set = numpy_dataset

    def shuffle(self):
        # random shuffle of the data
        np.random.shuffle(self.data_set)
        return self

    def size(self):
        # returns the size of the data
        return np.size(self.data_set, 0)

    def batch(self,batch_size):
        # returns a generator
        left_index = 0
        right_index = batch_size

        while right_index < self.size():
            yield self.data_set[left_index:right_index]
            left_index = right_index
            right_index = left_index + batch_size
        
        
        yield self.data_set[left_index:self.size()]
    
    def split(self):
        ''' returns the training set and the validation set'''
        size = self.size()
        splitting_index = int(size * (2 / 3))
        training = Dataset(self.data_set[0:splitting_index])
        validation = Dataset(self.data_set[splitting_index:size])
        return training, validation
               
        

class MonkDataset:
    
    @staticmethod
    def load(datapath):
        dataset = []
        with open(datapath, 'r') as fp:
            for line in fp:
                words_list = line.strip().split()
                id = words_list[7]
                output_class = int(words_list[0])
                inputs = list(map(int,words_list[1 : 7]))
                inputs = MonkDataset.__encode_1ofk(inputs)
                dataset.append((id,inputs, [output_class]))

        return Dataset(np.array(dataset))
    
    @staticmethod
    def __encode_1ofk(inputs):
        max_values = [3,3,2,3,4,2]
        encoded_inputs = []
        for i in range(len(inputs)):
            for j in range(max_values[i]):
                if j + 1 == inputs[i]:
                    encoded_inputs.append(1)
                else:
                    encoded_inputs.append(0)
        return encoded_inputs
    


