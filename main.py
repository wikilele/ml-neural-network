import neuralnetwork as nn
import datasets as ds


def main():

    dataset = ds.load('monk1.train')
    dataset.shuffle() # shuffles the dataset
    dataset.size()
    dataset.batch(2) # returning an iterator which produced batches of  size 2

    nn.input_layer(17)
    nn.hidden_layer(3)
    nn.hidden_layer(3)
    nn.output_layer(2)
    model = nn.build()

    model.fit(dataset.batch(2), epochs=300)

    model.infer('new data')

    model.evaluate(dataset.batch(2))



if __name__ == '__main__':
    main()