from itertools import product
import numpy as np
from .metrics import Metrics

__models_index = 0
__models = {}

training_set = None
validation_set = None


def add_model(model):
    global __models_index
    global __models
    __models_index += 1
    __models[__models_index] = {}
    __models[__models_index]['model'] = model
    __models[__models_index]['training_metrics'] = Metrics()
    __models[__models_index]['validation_metrics'] = Metrics()

    return __models_index

def get_model(model_index):
    return __models[model_index]['model']

def models():
    global __models
    for k  in __models.keys():
        yield k, __models[k]['model']

def set_datasets(trainset, valset):
    global training_set
    global validation_set
    training_set = trainset
    validation_set = valset

def compute_error(model_id, train_outs, val_outs, metrics=['mse']):
    global __models
    global training_set
    global validation_set

    __models[model_id]['training_metrics'].compute_error(train_outs, training_set, metrics=metrics)
    __models[model_id]['validation_metrics'].compute_error(val_outs, validation_set, metrics=metrics)

def compute_other(model_id, train_outs, val_outs, metrics=['acc'], threshold=0):
    global __models
    global training_set
    global validation_set

    __models[model_id]['training_metrics'].compute_other(train_outs, training_set, metrics=metrics,threshold=threshold)
    __models[model_id]['validation_metrics'].compute_other(val_outs, validation_set, metrics=metrics, threshold=threshold)

def avg_mse():
    global __models

    tr_erros = []
    val_errors = []
    for m_info in __models.values():
        tr_erros.append(m_info['training_metrics'].mse)
        val_errors.append(m_info['validation_metrics'].mse)
    
    return arrays_mean(tr_erros), arrays_mean(val_errors)

def avg_acc():
    global __models
    tr_acc = []
    val_acc = []
    for m_info in __models.values():
        tr_acc.append(m_info['training_metrics'].acc)
        val_acc.append(m_info['validation_metrics'].acc)
    
    return arrays_mean(tr_acc), arrays_mean(val_acc)

def get_metrics():
    return Metrics()



# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
def grid_search(param_grid):
    # the function works with lists
    if not isinstance(param_grid,list):
        param_grid = [param_grid]
    
    for p in param_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params

def arrays_mean(array_list):
    ''' 
    takes a matrix of loats and produces a list with the mean of the values in the provided elements
    input [ [1,2,3], [2,3,4] ]
    output [ 1.5, 2.5, 3.5 ]
    '''
    mean = np.array(array_list[0])
    for o in array_list[1:]:
        mean = np.add(mean, np.array(o))
    
    mean = np.divide(mean, len(array_list))
    return mean.tolist()