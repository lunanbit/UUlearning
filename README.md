# Keras implementation of UU learning

This is a reproducing code for the ICLR'19 paper: On the Minimal Supervision for Training Any Binary Classifier from Only Unlabeled Data.

* ```loss.py``` has a keras implementation of the risk estimator for UU learning (see Eq.(10) in the paper) and its simplified version (see Eq.(12) in the paper).

* ```experiment.py``` is an example code of UU learning. 

Datasets are MNIST preprocessed in such a way that even digits form the P class and odd digits form the N class and
CIFAR10 preprocessed in such a way that the P class is composed of `bird’, `cat’, ‘deer’, ‘dog’, ‘frog’ and ‘horse’; the N class is composed of ‘airplane’, ‘automobile’, ‘ship’ and ‘truck’.

## Requirements
* Python 3
* Numpy 1.14.1
* Keras 2.1.4
* Tensoflow 1.8.0
* Scipy 1.0.0
* Matplotlib 2.1.2
  

## Quick start
You can run an example code of UU learning on benchmark datasets (MNIST, CIFAR-10).

    python experiment.py --dataset mnist --mode UU

You can see additional information by adding ```--help```.

## Result
After running ```experiment.py```, the test figure and log file are made in ```output/dataset/``` by default.
The errors are measured by zero-one loss.
