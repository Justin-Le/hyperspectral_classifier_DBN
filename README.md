Notes on the dataset:

The Pavia Centre image and its groundtruth in .txt format are located in the Data directory.

Convert_mattotxt was used to convert the original .mat image and ground truth to .txt files. The original .mat files can be obtained from the [University of Palis Vasco] (http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes). 

The complete Pavia Centre image cannot be loaded on some machines due to memory restrictions. To avoid crashes, an eighth-image is loaded by default.

####################

Notes on usage:

DBN_writeparams is intended to replace DBN.

The parameters of the DBN object should match in DBN_writeparams and DBN_example (e.g., n_ins, hidden_layers_sizes, n_outs).

The number of numpy.savetxt lines for weights should match the number of elements in hidden_layers_sizes. Same for biases. Same for numpy.loadtxt in DBN_example.

####################

To do:

Create a module for loading trained weights and biases.
Create eighths of the ground truth .txt file.
