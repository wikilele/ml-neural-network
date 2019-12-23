import neuralnetwork as nn
from neuralnetwork.metrics import Metrics
import datasets as ds
from utils import *
import os
import json

def main():
    if not os.path.isdir('./plots'):
        os.mkdir('./plots')

    trainset = ds.load('monks-1.train')
    validationset = ds.load('monks-1.test')
    
    param_grid = {
        'epochs' : [390],
        'weights_bound' : [0.00009],
        'learning_rate' : [0.09,0.2],
        'batch_size' : [trainset.size()]
    }
    
    grid_search_resutls = {}
    result_index = 0
    for params in grid_search(param_grid):
        print(params)
        epochs = params['epochs']
        batch_size = params['batch_size']
        weights_bound = params['weights_bound']
        learning_rate = params['learning_rate']
               

        nn.input_layer(17)
        nn.hidden_layer(3, activation='sigmoid')
        nn.output_layer(1, activation='sigmoid')
        nn.init_weights_random(weights_bound)
        nn.learning_rate(learning_rate,0)
        model = nn.build()

        train_metrics = Metrics()
        val_metrics = Metrics()
        for m in model.fit(trainset,batch_size,epochs):
            train_metrics.compute_error(m,trainset)
            val_metrics.compute_error(m,validationset)

        val_metrics.compute_other(model,validationset,threshold=0.5)  

        print("ACCURACY: " + str(val_metrics.acc) + " %")
        print("MSE: " + str(val_metrics.mse[-1]))    

        plt = plot_error(range(epochs),train_metrics.mse, val_metrics.mse)
        path = './plots/result' + str(result_index) + '.png'
        result_index +=1
        plt.savefig(path)
        plt.clf()
        grid_search_resutls[val_metrics.mse[-1]] = { 'params' : params, 'plotpath' : path}

    with open('grid_results.json','w+') as f:
        f.write(json.dumps(grid_search_resutls,indent=4, sort_keys=True))

if __name__ == '__main__':
    main()