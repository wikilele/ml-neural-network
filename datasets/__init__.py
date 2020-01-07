from .dataset import MonkDataset, CupDataset

def load(data_path, dtype):
    print("loading the dataset " + data_path + "...")
    # initialize and returns a dataset class
    if dtype == 'CUP':
        # ML CUP 
        dataset = CupDataset.load(data_path)
    elif dtype == 'monks': 
        dataset = MonkDataset.load(data_path)
    return dataset
