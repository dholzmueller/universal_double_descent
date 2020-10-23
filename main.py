# Copyright 2020 The universal_double_descent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from plotting import *


if __name__ == '__main__':
    # compute empirical curves (saves results, does not recompute results if saved results already exist)
    # will compute n_rep times 1000 Monte-Carlo results and save every 1000 results.
    compute_random_nn_results(layer_sizes=[10, 256, 256, 256], n_rep=10)
    # relu is very noisy, needs more repetitions
    compute_random_nn_results(layer_sizes=[10, 256, 256, 256], n_rep=100, act_names=['ReLU'])
    # plot using the sphere and optimized feature maps
    # is very zoomed-in and needs more repetitions for small estimation error
    compute_sphere_results(n_rep=100)
    compute_best_nn_results(n_train=15, n_rep=100, hidden_sizes=[256, 256])
    compute_best_nn_results(n_train=60, n_rep=100, hidden_sizes=[256, 256])
    # ntk results are also noisy and need more repetitions
    compute_random_ntk_results(layer_sizes=[4, 6, 1], n_rep=100)  # NTK has 30-dimensional feature map (4*6 + 6*1 = 30)
    compute_poly_kernel_results(n_rep=10)
    compute_rff_results(n_rep=10)
    compute_rff_results(n_rep=100, weight_gain=1/30)

    # plot figures based on saved results
    plot_nn()
    plot_poly_kernel()
    plot_ntk()
    plot_rff()
    plot_rff_vs_sphere()
    plot_lower_bound_hypotheses()
    plot_counterexample()

    # experimentally verify (FRK) for NTK fetaure maps
    # as described in the paper via computing the maximum "inverse condition number"
    verify_frk_for_ntk(layer_sizes=[4, 6, 1])

    # the following code was used to check whether the formula for the sphere is symmetric
    # it turns out that the formula with switched n, p does not exactly hold for the underparameterized case
    # compute_sphere_results(n_rep=1000, p=2, n=2, max_n=20)
    # plot_lower_bound_hypotheses_2()

    # the following code was used to verify that the implemented polynomial kernel feature map
    # yields the polynomial kernel
    # test_poly_kernel()

    # the following code was used to verify that the NTK implementation is correct
    # test_ntk()
