import numpy

imraw = numpy.loadtxt('pavia_centre_image_quarter.txt')
gtraw = numpy.loadtxt('pavia_centre_groundtruth.txt')

val_idx = 0
example_size = 102
imlist = []

while val_idx < len(imraw):
  imlist.append(imraw[val_idx : val_idx + example_size])
  val_idx = val_idx + example_size

imarray = numpy.array(imlist, dtype='float32')
gtarray = numpy.array(gtraw, dtype='int64')

test_set = (imarray, gtarray)
test_set
