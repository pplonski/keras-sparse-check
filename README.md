# keras-sparse-check
This code is for comparison of training on dense vs sparse data used in neutrino experiments. In `common.py` file there are:
 - code that reads data into dense or sparse format. In case of dense format data is stored in `numpy` array. For sparse representation data is stored as list of sparse `scipy` matrices
 - code that creates network
 - sparse data generator - it generates batches with dense data based on list of sparse matrices

To run comparison please run:
 - for training with dense data `python train_dense.py`
 - for training with sparse data `python train_sparse.py`

### Prerequisites
 1. You have installed `keras`, `numpy`, `scipy`.
 2. Please provide proper paths to your data in functions: `get_y_data`, `get_x_data_sparse`, `get_x_data_dense` in `common.py` file. This example is for data with `3` classes and images size is `32x32`	with depth `1`. 
