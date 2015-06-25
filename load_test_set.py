import numpy

# Load image and ground truth from
# text files that are delimited by whitespace
imraw = numpy.loadtxt('pavia_centre_image_quarter.txt')
gtraw = numpy.loadtxt('pavia_centre_groundtruth.txt')

val_idx = 0
example_size = 102
imlist = []

# imlist is a list of images,
# where each image is a list of values 
while val_idx < len(imraw):
  imlist.append(imraw[val_idx : val_idx + example_size])
  val_idx = val_idx + example_size

# Store images and ground truths as numpy arrays
# in order to use them in Theano
imarray = numpy.array(imlist, dtype='float32')
gtarray = numpy.array(gtraw, dtype='int64')

# Theano accepts an input as a set,
# where each set is a 2-tuple that
# contains the image array and ground truth array
test_set = (imarray, gtarray)

# print test_set
