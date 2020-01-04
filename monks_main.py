#!/usr/bin/python3.6
import sys
from statistics import mode

import datasets as ds
import matplotlib.pyplot as plt
import model_selection as ms
import neuralnetwork as nn
import results as res
from utils import printProgressBar
import json

def monks1(task_type, param_grid, model_assessment=False):
    # this file contains the whole dataset
    # we rely on it instead of using the provided splitting 
    # because in that way we simulate a splitting according to hold-out technique
    dataset = ds.load('datasets/'+ task_type + '.test', 'monks')
    dataset.shuffle() # bacause data are taken randomly in monks-1.train
    # simple hold-out strategy 
    # ~123 elements for training set as in the original splitting
    # validation set is half of training set
    trainvalset, testset = dataset.split(43/100)
    trainset, validationset = trainvalset.split(66.6/100)
    

    for params in ms.grid_search(param_grid):
        
        # if batch size is -1 means we want the batch equal to the entire training set size
        params['batch_size'] = params['batch_size'] if params['batch_size'] > 0 else trainset.size()
        print(params)
        
        # getting the parameter
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

        # getting the average of errors and accuracy         
        avg_tr_error, avg_val_error = ms.avg_mse()
        avg_tr_acc, avg_val_acc = ms.avg_acc()

        res.set_task('monks1')
        # printing the error
        plt.plot(range(epochs), avg_tr_error, '-', label='train', color='black')
        plt.plot(range(epochs), avg_val_error, '-', label='val', color='red')

        plt.xlabel('epochs') 
        plt.legend(loc='upper right') 
        plt.title('Mean Square Error') 
        msepath = res.save_plot(plt,'mse')

        '''
        # we might print also the accuracy graph, but it might be not really good looking
        plt.plot(range(epochs), avg_tr_acc, '-', label='train', color='black')
        plt.plot(range(epochs), avg_val_acc, '-', label='val', color='red')

        plt.xlabel('epochs') 
        plt.legend(loc='lower right') 
        plt.title('Accuracy') 
        res.save_plot(plt,'acc')
        '''

        # precision and recall will be used during model assessment (see below)
        final_accuracy = avg_val_acc[-1]

        # adding the result
        res.add_result(avg_tr_error[-1], params['batch_size'], params['weights_bound'], params['learning_rate'] , final_accuracy, msepath)

    res.add_result_header('mse','batch_s','weights', 'lr', 'acc', 'path')     
    res.save_results()
    
    # WARNING this code must be executed only once
    # it must be executed only after model selection otherwise we will invalidate the test set
    if model_assessment:
        # here we want to use the testset to assess the model performances
        trainied_models = [m  for _, m in  ms.models()]
        voted_outputs = []
        for batch in testset.batch(1):
            for pattern in batch:
                tmp_outputs = []
                for m in trainied_models:
                    tmp_outputs.append( m.classify(pattern[1],threshold=0.5) )
                
                # we get the most frequent element ( majority vote)
                voted_outputs.append(mode(tmp_outputs))

        metrics = ms.get_metrics()
        target_outputs = [ x[0] for x in testset.data_set[:,2]]
        # computing acc, rec and precision for the testset
        acc = metrics.accuracy(voted_outputs,target_outputs)
        recall = metrics.recall(voted_outputs, target_outputs)
        precision = metrics.precision(voted_outputs, target_outputs)
        
        print("ACCURACY " + str(acc))
        print("PRECISION " + str(precision))
        print("RECALL " + str(recall))


def usage_and_exit():
    print("USAGE AND EXIT")
    print("./monks_main.py [monks_task] [model_selection/paramgrid.json] [-a] ")
    print()
    print("monks_task can be monks-1, monks-2, monks-3")
    print("-a  should be given if we want model assessment at the end")
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
        
    monks1(task_type, param_grid, model_assessment=model_assessment)
