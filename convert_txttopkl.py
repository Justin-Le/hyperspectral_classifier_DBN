import cPickle
import numpy

import theano
import theano.tensor as T

def convert_txttopkl():

    eighth1 = numpy.loadtxt('pavia_centre_image_eighth.txt')
    eighth2 = numpy.loadtxt('pavia_centre_image_eighth2.txt')
    eighth3 = numpy.loadtxt('pavia_centre_image_eighth3.txt')
    eighth4 = numpy.loadtxt('pavia_centre_image_eighth4.txt')
    eighth5 = numpy.loadtxt('pavia_centre_image_eighth5.txt')
    eighth6 = numpy.loadtxt('pavia_centre_image_eighth6.txt')
    eighth7 = numpy.loadtxt('pavia_centre_image_eighth7.txt')      
    eighth8 = numpy.loadtxt('pavia_centre_image_eighth8.txt')      
    gt = numpy.loadtxt('pavia_centre_groundtruth.txt')
    
    f = file('pavia_centre.pkl', 'wb')

    for d in [eighth1, eighth2, eighth3, eighth4, eighth5, eighth6, eighth7, eighth8, gt]:
	cPickle.dump(d, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('pavia_centre.pkl', 'rb')

    # list of eighths of the image
    image = []
    # The first 8 elements of the pickled file
    # are the 8 eighths of the image
    for i in range(8):
	image.append(cPickle.load(f))
     
    # The last element of the pickled file
    # is the complete ground truth
    groundtruth = cPickle.load(f)
    f.close()

    # list of images
    # where each image has 102 values
    imlist = []
    for i in xrange(len(image)):
	for j in xrange(len(image[i]) / 102):
	    imlist.append(image[i][102*j : 102*(j+1)])

    # imlist contains 1096*715 = 783,640 images

    # Store images and ground truths as numpy arrays in shared variables
    # in order to use them in Theano
    borrow = True
    train_set_x = theano.shared(numpy.asarray(imlist[0:583640],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    train_set_y = T.cast(
			    theano.shared(numpy.asarray(groundtruth[0:583640],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)
    test_set_x = theano.shared(numpy.asarray(imlist[583640:683640],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    test_set_y = T.cast(
			    theano.shared(numpy.asarray(groundtruth[583640:683640],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
		       )
    valid_set_x = theano.shared(numpy.asarray(imlist[683640:783640],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    valid_set_y = T.cast(
			    theano.shared(numpy.asarray(groundtruth[683640:783640],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)

    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	(test_set_x, test_set_y)]

    # Store as shared variables in order to print dimensions
    train_set_y_shared = theano.shared(numpy.asarray(groundtruth[0:583640]))
    valid_set_y_shared = theano.shared(numpy.asarray(groundtruth[583640:683640]))
    test_set_y_shared = theano.shared(numpy.asarray(groundtruth[683640:783640]))

    # Check that the dimensions of the sets are equal to those specified above
    print (('\nDimensions of training set (pixels by bands-per-pixel): %s\n') % ((train_set_x.get_value().shape,)))
    print (('\nDimensions of validation set (pixels by bands-per-pixel): %s\n') % ((valid_set_x.get_value().shape,)))
    print (('\nDimensions of test set (pixels by bands-per-pixel): %s\n') % ((test_set_x.get_value().shape,)))
    print (('\nDimensions of ground truth of training set (number of pixel labels): %s\n') % ((train_set_y_shared.get_value().shape,)))
    print (('\nDimensions of ground truth of validation set (number of pixel labels): %s\n') % ((valid_set_y_shared.get_value().shape,)))
    print (('\nDimensions of ground truth of test set (number of pixel labels): %s\n') % ((test_set_y_shared.get_value().shape,)))

    return datasets, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

if __name__ == '__main__':
    convert_txttopkl() 
