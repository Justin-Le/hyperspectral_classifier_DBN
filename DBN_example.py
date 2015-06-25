import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from DBN import DBN

def test_DBN_example(dataset='mnist.pkl.gz', batch_size=10):
    """"
    Tests a trained DBN on a set of images.

    For each batch:
    Given an image x, this function extracts the probability
    that x belongs to each class 
    and the predicted class to which x belongs
    (the class whose probability is highest given x)

    :type dataset: string
    :param dataset: path to the pickled data
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """"

    datasets = load_data(dataset)

    example_set_x, example_set_y = datasets[0]

    n_batches = example_set_x.get_value(borrow=True).shape[0] / batch_size 

    print(('number of batches: %i') % (n_batches))

    numpy_rng = numpy.random.RandomState(123)
    n_ins = 28*28
    hidden_layers_sizes = [10, 10, 10]
    n_outs = 10

    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_outs)

    print 'Loading model . . . '
    
    # If the text file is a list of weights,
    # delimited by whitespace,
    # then loaded_weights[i] is a numpy array of floats
    loaded_weights = [numpy.asarray([0])] * 3
    loaded_weights[0] = numpy.loadtxt('weights_layer0.txt')
    loaded_weights[1] = numpy.loadtxt('weights_layer1.txt')
    loaded_weights[2] = numpy.loadtxt('weights_layer2.txt')

    # print loaded_weights

    # As above, loaded_biases[i] is a numpy array of floats
    loaded_biases = [numpy.asarray([0])] * 3
    loaded_biases[0] = numpy.loadtxt('biases_layer0.txt')
    loaded_biases[1] = numpy.loadtxt('biases_layer1.txt')
    loaded_biases[2] = numpy.loadtxt('biases_layer2.txt')

    # Note: Use DBN_writeparams to obtain the above text files after training
           
    for layer_index in xrange(0,len(hidden_layers_sizes)):
        dbn.sigmoid_layers[layer_index].W.set_value(loaded_weights[layer_index])
        dbn.sigmoid_layers[layer_index].b.set_value(loaded_biases[layer_index])

        print (('Dimensions of layer %i weights: ') % (layer_index))
        print loaded_weights[layer_index].shape
        print (('Dimensions of layer %i biases: ') % (layer_index))
        print loaded_biases[layer_index].shape

        print (('Layer %i weights: ') % (layer_index)) 
        print loaded_weights[layer_index]
        print (('Layer %i biases: ') % (layer_index)) 
        print loaded_biases[layer_index]
       
    # start-snippet-2
     
    # example index
    index = T.lscalar('index')

    extract_prob = theano.function(
            [index],
            dbn.logLayer.p_y_given_x,
            givens={
                dbn.x: example_set_x[index*batch_size : (index+1)*batch_size]            
            }
        )

    extract_pred = theano.function(
            [dbn.logLayer.p_y_given_x],
            dbn.logLayer.y_pred            
        )

    test_score = 0.

    start_time = time.clock()

    for batch_index in xrange(n_batches):
                    y_prob = extract_prob(batch_index)
		    y_pred = extract_pred(y_prob)
                    # test_losses = test_model()
                    # test_score = numpy.mean(test_losses)
                    # print(('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))
        
                    if batch_index%1000 == 0:
  		        print(y_prob, y_pred)

    end_time = time.clock()
    print(
        (
            'with test performance %f %%'
        ) % (test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


if __name__ == '__main__':
    test_DBN_example()
