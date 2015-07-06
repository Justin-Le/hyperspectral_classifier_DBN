### Notes on the dataset:

* A simple script was used to convert the original .mat image and ground truth to .txt files. The Data directory contains examples of these .txt files (but not all). The original .mat files can be obtained from the [University of Palis Vasco] (http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes). 

### Notes on usage:

* DBN_writeparams is intended to replace DBN. DBN is only used by DBN_example.

* The parameters of the DBN object should match in DBN_writeparams and DBN_example (e.g., n_ins, hidden_layers_sizes, n_outs).

### To do:

* Create a module for loading trained weights and biases in DBN_example.
