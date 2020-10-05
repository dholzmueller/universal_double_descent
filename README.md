# On the Universality of the Double Descent Peak in Ridgeless Regression
This code can be used to reproduce figures from the paper "On the Universality of the Double Descent Peak in Ridgeless Regression". It is licensed under the Apache 2.0 license. The computations are mainly based on PyTorch, hence installing torch is a requirement. NumPy and Matplotlib are also required. Computations are configured to be run on a CPU since SVD computations on a GPU have been slow in our experiments.
Besides producing plots, the provided code includes the computational verification of the full-rank property (FRK) for n=p for analytic functions whose input has a Lebesgue density, in particular for NTK feature maps. For more details, see the comments in `main.py`.
The functionality is distributed as follows:
- `main.py` initiates all relevant computations. The computations take around one day on a 6-core CPU. The number of repetitions can be reduced in order to make computations faster (but less accurate). Already computed data will not be recomputed.
- `sampling.py` contains functionality to sample data from distributions and (random) feature maps.
- `computation.py` contains functionality to compute noise-error estimates using Monte Carlo. It also provides functions to compute and save the data for the specific (random) feature maps and input distributions used in the paper and implemented in `sampling.py`.
- `plotting.py` contains code that takes the saved data and generates the plots in the paper.
