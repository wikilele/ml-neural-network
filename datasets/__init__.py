from .dataset import MonkDataset

def load(data_path, dtype):
    print("loading the dataset " + data_path + "...")
    # initialize and returns a dataset class
    if dtype == 'cup':
        # ML CUP 
        raise NotImplementedError
    elif dtype == 'monks': 
        dataset = MonkDataset.load(data_path)
    return dataset
