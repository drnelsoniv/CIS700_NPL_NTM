# CIS700_NPL_NTM

This repository contains a semester project report, and associated sample code, towards implementing a Neural Turing Machine (NTM).

## Prerequisites
- Python https://www.python.org/
- PyTorch https://pytorch.org/

Note that I used Anaconda (https://www.anaconda.com/) to manage my Python/PyTorch installation.  The code provided was tested and working with Python v3.8.3 and PyTorch 1.4.0.

# Running
1. Download `main.py` and `ntm.py`.
2. Execute `main.py`.

Following execution, the code runs through the in-class example responsible for performing gradient descent to arrive at a specific memory location.
There are functions implemented for other aspects of the NTM, such as performing a read, write, and modifying the memory location weight matrix according to content-based lookup, performing interpolation with the previous weight matrix, performing a convolutional shift of the matrix, and finally sharpening the resulting weight matrix.  However, these were not integrated into an LSTM controller, so they are presently unused.



Next steps for this project are to take the component pieces implemented and integrate them into an LSTM controller to produce the NTM.
