#!/usr/bin/python3.6
import json
import sys
import time
from statistics import mean

import datasets as ds
import model_selection as ms
import neuralnetwork as nn
import numpy as np
import results as res
from datasets.dataset import Dataset
from utils import printProgressBar as ppb


def cup(param_grid):
    dataset = ds.load('datasets/ML-CUP19-TR.csv','CUP')
    # we do the train combining the previous trainingset and validation set
    # to have more data
    trainset, testset = dataset.split(75/100) 
    # data normalization
    
    params = next(ms.grid_search(param_grid))
    print(params)
    params['batch_size'] = params['batch_size'] if params['batch_size'] > 0 else trainset.size()
        
    epochs = params['epochs']
    batch_size = params['batch_size']

    runs_number = 1
    for run in range(runs_number):
        nn.from_parameters(params, 'sigmoid', 'linear')
        model = nn.build()
        ms.add_model(model)

    ms.set_datasets(trainset, testset)
        
    start = time.time()
    for e in range(epochs):
        ppb(e+1, epochs, prefix='Training', suffix='Completed')
        for model_id, model in ms.models():
            model.fit(trainset, batch_size, e)
                    
            train_outputs = model.forward_dataset(trainset)
            test_outputs = model.forward_dataset(testset)

        ms.compute_error(model_id, train_outputs, test_outputs, metrics=['mse','mee'])

    training_time = time.time() - start
    print('TRAINING TIME: ' +str(training_time)+ 'seconds')

    avg_tr_mse, avg_ts_mse = ms.avg_mse()
    avg_tr_mee, avg_ts_mee = ms.avg_mee()

    res.set_task('CUP')
    plt = res.plot_mse(epochs, avg_tr_mse, avg_ts_mse, params, label2='test')
    msepath = res.save_plot(plt, 'mse')

        
    plt = res.plot_mee(epochs, avg_tr_mee, avg_ts_mee, params, label2='test')
    res.save_plot(plt, 'mee')

    print("TRAINING MSE " + str(avg_tr_mse[-1]))
    print("TRAINING MEE " + str(avg_tr_mee[-1]))


    # here we want to use the testset to assess the model performances
    trained_models = [m  for _, m in  ms.models()]
    avg_outputs = []
    for batch in testset.batch(1):
        for pattern in batch:
            tmp_real_outputs_x = []
            tmp_real_outputs_y = []
            for m in trained_models:
                real_out = m.feed_forward(pattern[1])
                tmp_real_outputs_x.append(real_out[0])
                tmp_real_outputs_y.append(real_out[1])

                # we get the average output to compute the error
                avg_outputs.append([mean(tmp_real_outputs_x), mean(tmp_real_outputs_y)])

    metrics = ms.get_metrics()
    mse = metrics.mean_square_error(avg_outputs, testset.data_set[:,2])
    mee = metrics.mean_euclidian_error(avg_outputs, testset.data_set[:,2])

    print("MSE " + str(mse))
    print("MEE " + str(mee))

    blindds = ds.load_blind('datasets/ML-CUP19-TS.csv','CUP')

    avg_outputs = []
    for batch in blindds.batch(1):
        for pattern in batch:
            tmp_real_outputs_x = []
            tmp_real_outputs_y = []
            for m in trained_models:
                real_out = m.feed_forward(pattern[1])
                tmp_real_outputs_x.append(real_out[0])
                tmp_real_outputs_y.append(real_out[1])

                # we get the average output to compute the error
                avg_outputs.append([mean(tmp_real_outputs_x), mean(tmp_real_outputs_y)])

    with open("report/poxebur_wikilele_ML-CUP-TS.csv", "a+") as cupfile:
        # cleaning the file
        cupfile.seek(0)
        cupfile.truncate()

        cupfile.write("# Leonardo Frioli Luigi Quarantiello \n")
        cupfile.write("# poxebur_wikilele \n")
        cupfile.write("# ML-CUP19 \n")
        cupfile.write("# 10/01/2020 \n")

        for i in range(len(avg_outputs)):
            cupfile.write(str(i +1) + ", " + str(avg_outputs[i][0]) + ", " +  str(avg_outputs[i][1]) + "\n")

if __name__ == '__main__':
    if sys.argv[1]:
        try:
            with open(sys.argv[1],'r') as pgf:
                param_grid = json.load(pgf)
        except FileNotFoundError as e:
            print(e)
            exit(-1)

    cup(param_grid)
