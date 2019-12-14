import neuralnetwork as nn
import datasets as ds
import matplotlib.pyplot as plt 
from neuralnetwork.weights_service import WeightsService

def plot_error(epochs, error):
        plt.plot(epochs,error)

        plt.xlabel('epochs') 
        plt.ylabel('error') 
        plt.title('Mean Square Error graph') 
        plt.show() 

def main():

    dataset = ds.load('monks-1.train')
    dataset.shuffle() # shuffles the dataset
    # dataset.size()
    # dataset.batch(dataset.size()) # returning an iterator which produced batches of  size 2

    ws = WeightsService(-0.00009, 0.00009)

    nn.input_layer(17)
    nn.hidden_layer(3, activation='sigmoid')
    nn.output_layer(1, activation='sigmoid')
    nn.weights_service(ws)
    nn.learning_rate(0.7,320)
    model = nn.build()

    epochs = 320

    #metrics = model.fit(dataset,1,epochs)
    metrics = model.fit(dataset,dataset.size(),epochs)
    
    testset = ds.load('monks-1.test')
    classification_outputs = []
    for batch in testset.batch(1):
        for patter in batch:
            out = model.feed_forward(patter[1])
            if out[0] >= 0.5:
                classification_outputs.append(1)
            else:
                classification_outputs.append(0)

    target = [ x[0] for x in testset.data_set[:,2]]
    metrics.accuracy(classification_outputs, target)     

    print("ACCURACY: " + str(metrics.acc) + " %")
    print("MSE: " + str(metrics.mse[-1]))    

    plot_error(range(epochs), metrics.mse)



if __name__ == '__main__':
    main()