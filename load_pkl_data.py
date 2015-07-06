import cPickle
import numpy

def load_pkl_data():

    eighth1 = numpy.loadtxt('pavia_centre_image_eighth.txt')
    eighth2 = numpy.loadtxt('pavia_centre_image_eighth2.txt')
    eighth3 = numpy.loadtxt('pavia_centre_image_eighth3.txt')
    eighth4 = numpy.loadtxt('pavia_centre_image_eighth4.txt')
    eighth5 = numpy.loadtxt('pavia_centre_image_eighth5.txt')
    eighth6 = numpy.loadtxt('pavia_centre_image_eighth6.txt')
    eighth7 = numpy.loadtxt('pavia_centre_image_eighth7.txt')      
    gt = numpy.loadtxt('pavia_centre_groundtruth.txt')
    
    f = file('pavia_centre.pkl', 'wb')

    for d in [eighth1, eighth2, eighth3, eighth4, eighth5, eighth6, eighth7, gt]:
	cPickle.dump(d, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    f = file('pavia_centre.pkl', 'rb')

    image = []
    for i in range(7):
	image.append(cPickle.load(f))
    
    groundtruth = cPickle.load(f)
    f.close()

    return image, groundtruth

if __name__ == '__main__':
    load_pkl_data() 
