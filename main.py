import datasets as ds
import matplotlib.pyplot as plt
import model_selection as ms
import neuralnetwork as nn
import results as res
from utils import printProgressBar


def monks1():
    # this file contains the whole dataset
    # we rely on it instead of using the provided splitting 
    # because in that way we simulate a splitting according to hold-out technique
    dataset = ds.load('monks-1.test')
    dataset.shuffle() # bacause data are taken randomly in monks-1.train
    # simple hold-out strategy 
    # ~123 elements for training set as in the original splitting
    # validation set is half of training set
    trainvalset, testset = dataset.split(43/100)
    trainset, validationset = trainvalset.split(66.6/100)
    
    param_grid = {
        'weights_bound' : [0.00009],
        'learning_rate' : [0.14,0.09],
        'batch_size' : [trainset.size()]
    }
    
    for params in ms.grid_search(param_grid):
        print(params)
        epochs = 390 # value taken from the monks problem paper
        batch_size = params['batch_size']
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

        res.add_result(avg_tr_error[-1], params,metrics_values, msepath)
          
    res.save_results()
    

if __name__ == '__main__':
    monks1()
