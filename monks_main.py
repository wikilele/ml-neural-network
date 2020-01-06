#!/usr/bin/python3.6
import json
import sys
import time
from statistics import mode, mean

import datasets as ds
import matplotlib.pyplot as plt
import model_selection as ms
import neuralnetwork as nn
import results as res
from utils import printProgressBar


def monks(task_type, param_grid, model_assessment=False):
    # this file contains the whole dataset, we rely on it instead of using the provided splitting 
    # because in that way we simulate a splitting according to hold-out technique
    dataset = ds.load('datasets/'+ task_type + '.test', 'monks')
    dataset.shuffle() # bacause data are taken randomly in monks-*.train
    # simple hold-out strategy 
    # ~123 elements for training set as in the original splitting (monks-1, monks-3)
    splitting = 43/100
    if task_type == 'monks-2':
        # monks-2 uses ~169 elements in the training set
        splitting = 59/100

    trainvalset, testset = dataset.split(splitting)
    # validation set is half of training set
    trainset, validationset = trainvalset.split(66.6/100)
    

    for params in ms.grid_search(param_grid):
        
        # if batch size is -1 means we want the batch equal to the entire training set size
        params['batch_size'] = params['batch_size'] if params['batch_size'] > 0 else trainset.size()
        print(params)
        
        epochs = params['epochs'] # value taken from the monks problem paper
        batch_size = params['batch_size']

        # trying different runs, to be independent from random weights init
        # and to have a bias-variance estimation (ensemble learning) when using inference on testset   
        runs_number = 3 # 5 can be used as well
        for r in range(runs_number): 
            # we are going to init more instances of the model to 
            # perform a better computation of the metrics    
            nn.from_parameters(params, 'sigmoid', 'sigmoid')
            model = nn.build()

            ms.add_model(model) 
        
        ms.set_datasets(trainset,validationset)

        start_time = time.time()
        for e in range(epochs):
            printProgressBar(e + 1, epochs, prefix = 'Training:', suffix = 'Complete')

            # for each model we initialized above
            for model_id, model in ms.models():
                # doing one step of training
                model.fit(trainset,batch_size,e)
                    
                # computing the output values for this training step
                train_outputs = model.forward_dataset(trainset)
                val_outputs = model.forward_dataset(validationset)

                # compute the metrics
                ms.compute_error(model_id, train_outputs, val_outputs)
                ms.compute_other(model_id, train_outputs, val_outputs, metrics=['acc'],threshold=0.5)

        training_time = time.time() - start_time
        print("TRAINING TIME " + str(training_time) + " seconds") 

        # getting the average of errors and accuracy         
        avg_tr_error, avg_val_error = ms.avg_mse()
        avg_tr_acc, avg_val_acc = ms.avg_acc()
        # precision and recall will be used during model assessment (see below)
        final_accuracy = avg_val_acc[-1]

        res.set_task(task_type)

        plt = res.plot_mse(epochs, avg_tr_error, avg_val_error, params, final_accuracy)
        msepath = res.save_plot(plt,'mse')
        
        plt = res.plot_acc(epochs,avg_tr_acc,avg_val_acc,params)
        res.save_plot(plt,'acc')
        
        # adding the result
        res.add_result(avg_tr_error[-1], avg_val_error[-1], params['batch_size'], params['weights_bound'], params['learning_rate'] , params['momentum_alpha'], final_accuracy, msepath)
        
        if not model_assessment:
            # cleaning model selection for next run
            ms.clean()

    res.add_result_header('mse_tr' , 'mse_val','batch_s','weights', 'lr','m_alpha', 'acc', 'path')     
    res.save_results()
    
    # WARNING this code must be executed only once
    # it must be executed only after model selection otherwise we will invalidate the test set
    if model_assessment:
        # here we want to use the testset to assess the model performances
        trained_models = [m  for _, m in  ms.models()]
        voted_outputs = []
        avg_outputs = []
        for batch in testset.batch(1):
            for pattern in batch:
                tmp_voted_outputs = []
                tmp_real_outputs = []
                for m in trained_models:
                    class_out , real_out = m.classify(pattern[1],threshold=0.5)
                    tmp_voted_outputs.append( class_out )
                    tmp_real_outputs.append(real_out)
                
                # we get the most frequent element ( majority vote)
                voted_outputs.append(mode(tmp_voted_outputs))
                # we get the average output to compute the error
                avg_outputs.append([mean(tmp_real_outputs)])

        metrics = ms.get_metrics()
        target_outputs = [ x[0] for x in testset.data_set[:,2]]
        # computing acc, rec and precision for the testset
        acc = metrics.accuracy(voted_outputs,target_outputs)
        recall = metrics.recall(voted_outputs, target_outputs)
        precision = metrics.precision(voted_outputs, target_outputs)

        mse = metrics.mean_square_error(avg_outputs, testset.data_set[:,2])
        
        print("ACCURACY " + str(acc))
        print("PRECISION " + str(precision))
        print("RECALL " + str(recall))
        print("MSE " + str(mse))



def usage_and_exit():
    print("USAGE AND EXIT")
    print("./monks_main.py [monks_task] [model_selection/paramgrid.json] [-a] ")
    print()
    print("monks_task can be monks-1, monks-2, monks-3")
    print("-a  should be given if we want model assessment at the end")
    print("call with -a only if not performing grid search")
    print()
    exit(-1)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage_and_exit()
    
    task_type = ""
    if sys.argv[1] in ["monks-1","monks-2","monks-3"]:
        task_type = sys.argv[1]
    else:
        usage_and_exit()

    try:
        with open(sys.argv[2],'r') as pgf:
            param_grid = json.load(pgf)
    except FileNotFoundError as e:
        print(e)
        usage_and_exit()
    
    model_assessment = False
    if len(sys.argv) == 4:
        model_assessment = True
        
    monks(task_type, param_grid, model_assessment=model_assessment)
