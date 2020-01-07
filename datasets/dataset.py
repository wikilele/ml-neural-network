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
    
    def split(self, percentage):
        '''
        it splits the set according to the input parameter.
        it returns a two Dataset object:
        set1 is from 0 to size*percentage
        set2 is the remaining data
        '''
        size = self.size()
        splitting_index = int(size * percentage)
        set1 = Dataset(self.data_set[0:splitting_index])
        set2 = Dataset(self.data_set[splitting_index:size])
        
        return set1, set2

    def normalize(self):
        data_set = self.data_set.T

        for i,line in enumerate(self.data_set):
            # that is because line is a list of lists and line[0] are the IDs
            max_value = max(max(line[1]), max(line[2]))
            min_value = min(min(line[1]), min(line[2]))

            for i in range(1, 3):
                for j, elem in enumerate(line[i]):
                    line[i][j] = (elem - min_value)/(max_value - min_value)
        
                
    def print(self):
        print(self.data_set)

        

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
    
class CupDataset:

    @staticmethod
    def load(datapath):
        dataset = []
        with open(datapath, 'r') as file:
            for i,line in enumerate(file):
                targets = []
                words_list = line.strip().split(',')
                id = words_list[0]
                targets.append(float(words_list[-2]))
                targets.append(float(words_list[-1]))
                inputs = list(map(float,words_list[1:21]))

                dataset.append((id,inputs, targets))
    
        return Dataset(np.array(dataset))   



