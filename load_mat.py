import cPickle
import numpy

import theano
import theano.tensor as T
import scipy.io

def load_mat():
    data_dict = scipy.io.loadmat('train_set_x')
    data_val = data_dict['train_set_x']
    train_data_x = data_val.reshape((data_val.shape[1], data_val.shape[2]))

    data_dict = scipy.io.loadmat('train_set_y')
    data_val = data_dict['train_set_y']
    train_data_y = data_val.reshape((data_val.shape[1]))

    data_dict = scipy.io.loadmat('test_set_x')
    data_val = data_dict['test_set_x']
    test_data_x = data_val.reshape((data_val.shape[1], data_val.shape[2]))

    data_dict = scipy.io.loadmat('test_set_y')
    data_val = data_dict['test_set_y']
    test_data_y = data_val.reshape((data_val.shape[1]))

    data_dict = scipy.io.loadmat('dataset_x')
    data_val = data_dict['dataset_x']
    dataset_x = data_val.reshape((data_val.shape[1], data_val.shape[2]))

    data_dict = scipy.io.loadmat('dataset_y')
    data_val = data_dict['dataset_y']
    dataset_y = data_val.reshape((data_val.shape[1]))

    f = file('train_data_x.pkl', 'wb')
    cPickle.dump(train_data_x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('train_data_y.pkl', 'wb')
    cPickle.dump(train_data_y, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('test_data_x.pkl', 'wb')
    cPickle.dump(test_data_x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('test_data_y.pkl', 'wb')
    cPickle.dump(test_data_y, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('dataset_x.pkl', 'wb')
    cPickle.dump(dataset_x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('dataset_y.pkl', 'wb')
    cPickle.dump(dataset_y, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('train_data_x.pkl', 'rb')
    train_data_x = cPickle.load(f)
    f.close()

    f = file('train_data_y.pkl', 'rb')
    train_data_y = cPickle.load(f)
    f.close()

    f = file('test_data_x.pkl', 'rb')
    test_data_x = cPickle.load(f)
    f.close()

    f = file('test_data_y.pkl', 'rb')
    test_data_y = cPickle.load(f)
    f.close()

    f = file('dataset_x.pkl', 'rb')
    dataset_x = cPickle.load(f)
    f.close()

    f = file('dataset_y.pkl', 'rb')
    dataset_y = cPickle.load(f)
    f.close()

    # Store images and ground truths as numpy arrays in shared variables
    # in order to use them in Theano
    borrow = True
    train_set_x = theano.shared(numpy.asarray(train_data_x,
					   dtype=theano.config.floatX),
			     borrow=borrow)
    train_set_y = T.cast(
			    theano.shared(numpy.asarray(train_data_y,
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)
    valid_set_x = theano.shared(numpy.asarray(test_data_x,
					   dtype=theano.config.floatX),
			     borrow=borrow)
    valid_set_y = T.cast(
			    theano.shared(numpy.asarray(test_data_y,
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
			)
    test_set_x = theano.shared(numpy.asarray(dataset_x,
					   dtype=theano.config.floatX),
			     borrow=borrow)
    test_set_y = T.cast(
			    theano.shared(numpy.asarray(dataset_y,
					   dtype=theano.config.floatX),
			     borrow=borrow),
			    'int32'
		       )
    datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
	(test_set_x, test_set_y)]

    # Store as shared variables in order to print dimensions
    train_set_y_shared = theano.shared(numpy.asarray(train_data_y))
    valid_set_y_shared = theano.shared(numpy.asarray(test_data_y))
    test_set_y_shared = theano.shared(numpy.asarray(dataset_y))

    # Check that the dimensions of the sets are equal to those specified above
    print (('\nDimensions of training set (pixels by bands-per-pixel): %s\n') % ((train_set_x.get_value().shape,)))
    print (('\nDimensions of validation set (pixels by bands-per-pixel): %s\n') % ((valid_set_x.get_value().shape,)))
    print (('\nDimensions of test set (pixels by bands-per-pixel): %s\n') % ((test_set_x.get_value().shape,)))
    print (('\nDimensions of ground truth of training set (number of pixel labels): %s\n') % ((train_set_y_shared.get_value().shape,)))
    print (('\nDimensions of ground truth of validation set (number of pixel labels): %s\n') % ((valid_set_y_shared.get_value().shape,)))
    print (('\nDimensions of ground truth of test set (number of pixel labels): %s\n') % ((test_set_y_shared.get_value().shape,)))

    return datasets, train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y

if __name__ == '__main__':
    load_mat() 
