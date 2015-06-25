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
    """
    Tests a trained DBN on a set of images.

    For each batch:
    Given an image x, this function extracts the probability
    that x belongs to each class 
    and the predicted class to which x belongs
    (the class whose probability is highest given x)

    :type dataset: string
    :param dataset: path to the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_data(dataset)

    example_set_x, example_set_y = datasets[0]

    n_batches = example_set_x.get_value(borrow=True).shape[0] / batch_size 

    print(('number of batches: %i') % (n_batches))

    numpy_rng = numpy.random.RandomState(123)
    n_ins = 28*28
    hidden_layers_sizes = [1, 1, 1]
    n_outs = 10

    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_ins,
              hidden_layers_sizes=hidden_layers_sizes,
              n_outs=n_outs)

    print 'Loading model . . . '
    
    # If the text file is a list of weights,
    # delimited by whitespace,
    # then loaded_weights[i] is a numpy array of floats
    loaded_weights[0] = numpy.loadtxt('weights_layer0.txt')
    loaded_weights[1] = numpy.loadtxt('weights_layer1.txt')
    loaded_weights[2] = numpy.loadtxt('weights_layer2.txt')

    # Note: Use DBN_writeparams to obtain the above text files after training

    ############################################################
    # Load weights of first layer
    ############################################################

    # A list of matrices,
    # where each matrix represents the weights of one layer
    weight_matrices = [[0]] * len(hidden_layers_sizes)

    # The first layer has a weight matrix with dimensions of
    # (length of input image vector) x (width of first hidden layer)
    n_rows = n_ins
    n_cols = hidden_layers_sizes[0]

    for i in xrange(n_rows):
	# form a new row in the matrix
	row = []
	row.append(loaded_weights[0][i*n_cols : (i+1)*n_cols])
	
	# append the row to the matrix
	weight_matrices[0].append(row)
	
    ############################################################
    # Load weights of each layer after the first layer
    ############################################################

    for layer_index in xrange(1,len(hidden_layers_sizes)):
	n_cols = hidden_layers_sizes[layer_index]
	n_rows = hidden_layers_sizes[layer_index-1]

	for i in xrange(n_rows):
	    # form a new row in the matrix
	    row = []
	    row.append(loaded_weights[layer_index][i*n_cols : (i+1)*n_cols])

	    # append the row to the matrix
	    weight_matrices[layer_index].append(row)

    # Store weight_matrices as a numpy array in order to use it
    # in setting the value of the shared variables that hold
    # the weights in each layer
    weight_matrices_array = numpy.asarray(weight_matrices)

    for layer_index in xrange(0,len(hidden_layers_sizes)):
        dbn.sigmoid_layers[layer_index].W.set_value(weight_matrices_array[layer_index])
        
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
