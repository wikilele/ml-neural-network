import random

class Dataset():

    def __init__(self, datapath):
        print("loading the dataset ...")
        self.data_set = self._load(datapath) 

    def shuffle(self):
        # random shuffle of the data
        random.shuffle(self.data_set)
        return self

    def size(self):
        # returns the size of the data
        return len(self.data_set)

    def batch(self,batch_size):
        # returns a generator
        left_index = 0
        right_index = batch_size

        while right_index < self.size():
            yield self.data_set[left_index:right_index]
            left_index = right_index
            right_index = left_index + batch_size
        
        
        yield self.data_set[left_index:self.size()]
        
               

    def _load(self,datapath):
        # this method must be implemented in the sublcasses
        # it must return something like: [(id , [input1, input2, ...] , [ouput1,output2, ...])]
        return []
        

class MonkDataset(Dataset):

    def __init__(self,datapath):
        Dataset.__init__(self,datapath)
    
    def _load(self,datapath):
        dataset = []
        with open(datapath, 'r') as fp:
            for line in fp:
                words_list = line.strip().split()
                id = words_list[7]
                output_class = int(words_list[0])
                inputs = list(map(int,words_list[1 : 7]))
                inputs = self.__encode_1ofk(inputs)
                dataset.append((id,inputs, [output_class]))
        return dataset
    
    
    def __encode_1ofk(self,inputs):
        max_values = [3,3,2,3,4,2]
        encoded_inputs = []
        for i in range(len(inputs)):
            for j in range(max_values[i]):
                if j + 1 == inputs[i]:
                    encoded_inputs.append(1)
                else:
                    encoded_inputs.append(0)
        return encoded_inputs

