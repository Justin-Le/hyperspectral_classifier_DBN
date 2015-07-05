import numpy

import theano
import theano.tensor as T

def load_txt_data():
    print '\nLoading the image to be classified . . .\n'

    # Load images and ground truth from
    # text files that are delimited by whitespace
    imraw = numpy.loadtxt('pavia_centre_image_eighth.txt')[0:7293000]
    gtraw = numpy.loadtxt('pavia_centre_groundtruth.txt')[0:71500]

    print '\nDimensions of loaded image set: %s\n' % (imraw.shape,)
    print '\nDimensions of loaded ground truth: %s\n' % (gtraw.shape,)

    val_idx = 0
    example_size = 102
    imlist = []

    # imlist is a list of images,
    # where each image is a list of values 
    while val_idx < len(imraw):
      imlist.append(imraw[val_idx : val_idx + example_size])
      val_idx = val_idx + example_size

    print '\nNumber of loaded images: %s\n' % (len(imlist))

    # Store images and ground truths as numpy arrays in shared variables
    # in order to use them in Theano
    borrow = True
    train_set_x = theano.shared(numpy.asarray(imlist[0:51500],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    train_set_y = T.cast(
			    theano.shared(numpy.asarray(gtraw[0:51500],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)
    test_set_x = theano.shared(numpy.asarray(imlist[51500:61500],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    test_set_y = T.cast(
			    theano.shared(numpy.asarray(gtraw[51500:61500],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
		       )
    valid_set_x = theano.shared(numpy.asarray(imlist[61500:71500],
					   dtype=theano.config.floatX),
			     borrow=borrow)
    valid_set_y = T.cast(
			    theano.shared(numpy.asarray(gtraw[61500:71500],
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)

    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	(test_set_x, test_set_y)]

    # Store as shared variable in order to print dimensions
    train_set_y_shared = theano.shared(numpy.asarray(gtraw[0:51500]))

    # Check that the dimensions of an arbitrarily chosen set are equal to those specified above
    # (In this case, the training set was chosen.)
    print (('\nDimensions of image set (pixels by bands-per-pixel): %s\n') % ((test_set_x.get_value().shape,)))
    print (('\nDimensions of ground truth set (number of pixel labels): %s\n') % ((train_set_y_shared.get_value().shape,)))

    return datasets, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

if __name__ == '__main__':
    load_txt_data()
