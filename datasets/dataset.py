import random
import numpy as np

max_values = []
min_values = []

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

    def _normalize(self,data_index, trainset):
        # if data_index == 1 we consider the input patterns
        # if data_index == 2 we consider the output
        # for each column
        for i in range(len(self.data_set[:,data_index][0])):
            if trainset:
                # the first value of the tmp varaibles is setted in this way
                # cause we can't assume an upper or lower bound
                tmp_max = self.data_set[:,data_index][0][i]
                tmp_min = self.data_set[:,data_index][0][i]
                # for each row we comput max and min of the row
                for j in range(len(self.data_set)):
                    tmp_max = max([tmp_max, self.data_set[:,data_index][j][i]])
                    tmp_min = min([tmp_min, self.data_set[:,data_index][j][i]])
                max_values.append(tmp_max)
                min_values.append(tmp_min)

            # update each element of column i
            for j in range(len(self.data_set)):
                self.data_set[:,data_index][j][i] = (self.data_set[:,data_index][j][i] - min_values[i]) / (max_values[i] - min_values[i])


    def normalize(self, trainset):
        self._normalize(1, trainset)
        self._normalize(2, trainset)

        
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



