import datasets as ds
import model_selection as ms
import neuralnetwork as nn
from utils import printProgressBar as ppb
import time
import sys
import numpy as np
import json
import results as res
from datasets.dataset import Dataset

def cup(param_grid):
    dataset = ds.load('datasets/ML-CUP19-TR.csv','CUP')
    trainset, validationset = dataset.split(60/100)

    for params in ms.grid_search(param_grid):
        params['batch_size'] = params['batch_size'] if params['batch_size'] > 0 else trainset.size()
        print(params)

        epochs = params['epochs']
        batch_size = params['batch_size']

        runs_number = 3
        for run in range(runs_number):
            nn.from_parameters(params, 'sigmoid', 'sigmoid')
            model = nn.build()
            ms.add_model(model)

        ms.set_datasets(trainset, validationset)
        
        start = time.time()
        for e in range(epochs):
            ppb(e+1, epochs, prefix='Training', suffix='Completed')
            
            for model_id, model in ms.models():
                model.fit(trainset, batch_size, e)
                
                train_outputs = model.forward_dataset(trainset)
                val_outputs = model.forward_dataset(validationset)

                ms.compute_error(model_id, train_outputs, val_outputs, metrics=['mse','mee'])

        training_time = time.time() - start
        print('TRAINING TIME: ' +str(training_time)+ 'seconds')

        avg_tr_mse, avg_val_mse = ms.avg_mse()
        #avg_tr_mee, avg_val_mee = ms.avg_mee()

        res.set_task('CUP')
        plt = res.plot_mse(epochs, avg_tr_mse, avg_val_mse, params, 0)
        msepath = res.save_plot(plt, 'mse')

        '''
        plt = res.plot_mee(epochs, avg_tr_mee, avg_val_mee, params)
        meepath = res.save_plot(plt, 'mee')
        '''

        res.add_result(avg_tr_mse[-1], avg_val_mse[-1], params['batch_size'], params['weights_bound'], params['learning_rate'] , params['momentum_alpha'], msepath)
        ms.clean()

        res.add_result_header('mse_tr' , 'mse_val', 'mee_tr', 'mee_val', 'batch_s','weights', 'lr','m_alpha', 'acc', 'path')     
        res.save_results()

   


if __name__ == '__main__':
    if sys.argv[1]:
        try:
            with open(sys.argv[1],'r') as pgf:
                param_grid = json.load(pgf)
        except FileNotFoundError as e:
            print(e)
            exit(-1)

    cup(param_grid)