#
#  just a bunch of functions and classes taken from internet
#
from itertools import product
import matplotlib.pyplot as plt 
import os
import json

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


class Results:
    def __init__(self):
        if not os.path.isdir('./plots'):
            os.mkdir('./plots')
        self.grid_search_resutls = {}
        self.result_index = 0

    def plot_error(self, epochs, train_error, val_error):
        plt.plot(epochs,train_error, '-', label='train', color='black' )
        plt.plot(epochs,val_error, '--', label='validation', color='black')

        plt.xlabel('epochs') 
        plt.legend(loc='upper right') 
        plt.title('Mean Square Error graph')     
    
    def show_plot(self):
        plt.show()

    def save_plot(self):
        path = './plots/result' + str(self.result_index) + '.png'
        self.result_index +=1
        plt.savefig(path)
        plt.clf()
        return path
    
    def add_result(self,mse, params, path):
        self.grid_search_resutls[mse] = {'params' : params, 'plotpath': path}
    
    def save_results(self):
        with open('grid_results.json','w+') as f:
            f.write(json.dumps(self.grid_search_resutls,indent=4, sort_keys=True))
    
