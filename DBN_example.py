import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from DBN_writeparams import DBN

def test_DBN_example(dataset='none', batch_size=10):
    """
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
    """

    if dataset == 'none':
	print '\nLoading the image to be classified . . .\n'

	# Load images and ground truth from
	# text files that are delimited by whitespace
	imraw = numpy.loadtxt('pavia_centre_image_quarter.txt')[0:274]
	gtraw = numpy.loadtxt('pavia_centre_groundtruth.txt')[0:274]

	val_idx = 0
	example_size = 102
	imlist = []

	# imlist is a list of images,
	# where each image is a list of values 
	while val_idx < len(imraw):
	  imlist.append(imraw[val_idx : val_idx + example_size])
	  val_idx = val_idx + example_size

	# Store images and ground truths as numpy arrays in shared variables
	# in order to use them in Theano
        borrow = True
        example_set_x = theano.shared(numpy.asarray(imlist,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        example_set_y = T.cast(
                                  theano.shared(numpy.asarray(gtraw,
                                               dtype=theano.config.floatX),
                                 borrow=borrow),
                                  'int32'
                              )

        # store as a shared variable in order to print dimensions
	example_set_y_shared = theano.shared(numpy.asarray(gtraw))

        print (('\nDimensions of image set (pixels by bands-per-pixel): %s\n') % ((example_set_x.get_value().shape,)))
        print (('\nDimensions of ground truth set (number of pixel labels): %s\n') % ((example_set_y_shared.get_value().shape,)))

    else:
	datasets = load_data(dataset)

	example_set_x, example_set_y = datasets[0]


    n_batches = example_set_x.get_value(borrow=True).shape[0] / batch_size 
    print(('\nnumber of batches: %i\n') % (n_batches))

    # Create the DBN object
    numpy_rng = numpy.random.RandomState(123)
    n_ins = 102
    hidden_layers_sizes = [30, 30]
    n_outs = 5

    dbn = DBN(numpy_rng=numpy_rng, n_ins=102,
              hidden_layers_sizes=[30, 30],
              n_outs=5)

    print '\nLoading weights and biases . . .\n'
    
    # If the text file is a list of weights,
    # delimited by whitespace,
    # then loaded_weights[i] is a numpy array of floats

    # NOTE: CHANGE THE MULTIPLE OF loaded_weights TO THE NUMBER OF txt FILES LOADED
    # i.e., if loaded_weights[k] exists, then the line should read:
    # loaded_weights = [numpy.asarray([0])] * k

    loaded_weights = [numpy.asarray([0])] * 2
    loaded_weights[0] = numpy.loadtxt('weights_layer0.txt')
    loaded_weights[1] = numpy.loadtxt('weights_layer1.txt')
    # loaded_weights[2] = numpy.loadtxt('weights_layer2.txt')
    # Add as many layers as used in training
    
    # As above
    loaded_biases = [numpy.asarray([0])] * 2
    loaded_biases[0] = numpy.loadtxt('biases_layer0.txt')
    loaded_biases[1] = numpy.loadtxt('biases_layer1.txt')
    # loaded_biases[2] = numpy.loadtxt('biases_layer2.txt')

    # Note: Use DBN_writeparams to obtain the above text files
           
    for layer_index in xrange(0,len(hidden_layers_sizes)):
        dbn.sigmoid_layers[layer_index].W.set_value(loaded_weights[layer_index])
        dbn.sigmoid_layers[layer_index].b.set_value(loaded_biases[layer_index])

        print (('\nDimensions of layer %i weights:\n') % (layer_index))
        print loaded_weights[layer_index].shape
        print '\n'
        print (('\nDimensions of layer %i biases:\n') % (layer_index))
        print loaded_biases[layer_index].shape
        print '\n'

        print (('\nLayer %i weights:\n') % (layer_index)) 
        print loaded_weights[layer_index]
        print '\n'
        print (('\nLayer %i biases:\n') % (layer_index)) 
        print loaded_biases[layer_index]
        print '\n' 

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
  		        print '\nProbabilities of y given x:\n'
                        print y_prob 
                        print '\n'
  		        print '\nPredicted classes y:\n'
                        print y_pred
                        print '\n'

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
