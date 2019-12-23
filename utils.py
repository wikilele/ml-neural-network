#
#  just a bunch of functions and classes taken from internet
#
from itertools import product
import matplotlib.pyplot as plt 

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_error(epochs, train_error, val_error):
    plt.plot(epochs,train_error, 'b')
    plt.plot(epochs,val_error, 'r')

    plt.xlabel('epochs') 
    plt.ylabel('train/val error') 
    plt.title('Mean Square Error graph') 
    # plt.show() 
    
    return plt

    

# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
def grid_search(param_grid):
    
    # Always sort the keys of a dictionary, for reproducibility
    items = sorted(param_grid.items())
    if not items:
        yield {}
    else:
        keys, values = zip(*items)
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params