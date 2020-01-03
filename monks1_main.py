import sys
from statistics import mode

import datasets as ds
import matplotlib.pyplot as plt
import model_selection as ms
import neuralnetwork as nn
import results as res
from utils import printProgressBar
import json

def monks1(param_grid, model_assessment=False):
    # this file contains the whole dataset
    # we rely on it instead of using the provided splitting 
    # because in that way we simulate a splitting according to hold-out technique
    dataset = ds.load('datasets/monks-1.test', 'monks')
    dataset.shuffle() # bacause data are taken randomly in monks-1.train
    # simple hold-out strategy 
    # ~123 elements for training set as in the original splitting
    # validation set is half of training set
    trainvalset, testset = dataset.split(43/100)
    trainset, validationset = trainvalset.split(66.6/100)
    

    for params in ms.grid_search(param_grid):
        print(params)
        epochs = 390 # value taken from the monks problem paper
        # if batch size is -1 means we want the batch equal to the entire training set size
        batch_size = params['batch_size'] if params['batch_size'] > 0 else trainset.size()
        weights_bound = params['weights_bound']
        learning_rate = params['learning_rate']

        # trying different runs, to be independent from random weights init
        # and to have a bias-variance estimation       
        runs_number = 3 # 5 can be used as well
        models_id = []
        for r in range(runs_number): 
            # we are going to init more instances of the model to 
            # perform a better computation of the metrics 
            nn.input_layer(17)
            nn.hidden_layer(3, activation='sigmoid')
            nn.output_layer(1, activation='sigmoid')
            nn.init_weights_random(weights_bound)
            nn.learning_rate(learning_rate,0)
            model = nn.build()

            model_id = ms.add_model(model) 
            models_id.append(model_id)
        
        ms.set_datasets(trainset,validationset)

        for e in range(epochs):
            printProgressBar(e + 1, epochs, prefix = 'Training:', suffix = 'Complete')

            # for each model we initialized above
            for model_id in models_id:
                model = ms.get_model(model_id)
                # doing one step of training
                model.fit(trainset,batch_size,e)
                    
                # computing the output values for this training step
                train_outputs = model.forward_dataset(trainset)
                val_outputs = model.forward_dataset(validationset)

                # compute the metrics
                ms.compute_error(model_id, train_outputs, val_outputs)
                ms.compute_other(model_id, train_outputs, val_outputs, metrics=['acc'],threshold=0.5)

        # val_metrics.compute_other(model,validationset,metrics=['prec','rec'],threshold=0.5)   
        
        avg_tr_error, avg_val_error = ms.avg_mse()
        avg_tr_acc, avg_val_acc = ms.avg_acc()

        res.set_task('monks1')
        plt.plot(range(epochs), avg_tr_error, '-', label='train', color='black')
        plt.plot(range(epochs), avg_val_error, '-', label='val', color='red')

        plt.xlabel('epochs') 
        plt.legend(loc='upper right') 
        plt.title('Mean Square Error') 
        msepath = res.save_plot(plt,'mse')

        '''
        plt.plot(range(epochs), avg_tr_acc, '-', label='train', color='black')
        plt.plot(range(epochs), avg_val_acc, '-', label='val', color='red')

        plt.xlabel('epochs') 
        plt.legend(loc='lower right') 
        plt.title('Accuracy') 
        res.save_plot(plt,'acc')
        '''

        metrics_values =  {
            'accuracy': avg_val_acc[-1]
        #    'precision': val_metrics.prec,
        #    'recall': val_metrics.rec
        # this will be used in test phase
        }

        
        res.add_result(avg_tr_error[-1], params['batch_size'], params['weights_bound'], params['learning_rate'] , metrics_values['accuracy'], msepath)

    res.add_result_header('mse','batch_s','weights', 'lr', 'acc', 'path')     
    res.save_results()
    
    if model_assessment:
        # here we want to use the testset to assess the model performances
        trainied_models = ms.get_models()
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
        acc = metrics.accuracy(voted_outputs,target_outputs)
        recall = metrics.recall(voted_outputs, target_outputs)
        precision = metrics.precision(voted_outputs, target_outputs)
        
        print("ACCURACY " + str(acc))
        print("PRECISION " + str(precision))
        print("RECALL " + str(recall))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("USAGE AND EXIT")
        print("python3.6 monks1_main.py [model_selection/paramgrid.json] [-a] ")
        print()
        print("-a  should be given if we want model assessment at the end")
        exit(-1)
    
    with open(sys.argv[1],'r') as pgf:
        param_grid = json.load(pgf)
    
    model_assessment = False
    if len(sys.argv) == 3:
        model_assessment = True
        
    monks1(param_grid, model_assessment=model_assessment)
