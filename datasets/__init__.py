from .Dataset import MonkDataset

def load(data_path):
    # initialize and returns a dataset class
    if data_path.endswith('.csv'):
        # ML CUP 
        raise NotImplementedError
    elif data_path.startswith('monks'): 
        dataset = MonkDataset(data_path)
    return dataset
