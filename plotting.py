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

import matplotlib
#matplotlib.use('Agg')
matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}'
})
import matplotlib.pyplot as plt

from computation import *


def load_dd_results(name):
    # load all results computed for a configuration given by name
    # and collect them all into a single DoubleDescentResults object
    base_path = Path('data/double_descent') / name
    results = [utils.deserialize(file) for file in base_path.iterdir()]
    if len(results) == 0:
        raise RuntimeError(f'No result files for {name}')
    result = results[0]
    result.results_n = np.concatenate([r.results_n for r in results], axis=1)
    result.results_n_over = np.concatenate([r.results_n_over for r in results], axis=1)
    result.results_p = np.concatenate([r.results_p for r in results], axis=1)
    return result


def plot_counterexample():
    print('Plotting counterexample')
    np.random.seed(0)
    ns = list(range(1, 257))
    values = []
    stds = []
    n_mc = 1000000
    p = 30
    plt.figure(figsize=(5.5, 2.5))
    plt.axvline(p, color='#333333', linestyle='--', linewidth=1)
    for n in ns:
        print(f'n = {n}')
        samples = np.random.binomial(n, p=1/p, size=n_mc).astype(np.float64)
        samples[samples > 0] = 1./samples[samples > 0]
        values.append(np.mean(samples))
        stds.append(np.std(samples))
    values = np.array(values)
    # we want the estimated std of the mean estimator, not the estimated std of the data
    stds = np.array(stds)/(np.sqrt(n_mc - 1))
    plt.loglog(ns, [dd_lower_bound(n, p) for n in ns], 'k',  label='Lower bound', linewidth=1)
    plt.fill_between(ns, values-stds, values+stds, facecolor='#1f77b4', edgecolor=None, alpha=0.3)
    plt.loglog(ns, values, color='#1f77b4', label='Counterexample', linewidth=1)
    plt.grid(True, which='both')
    plt.xlabel('Number of points $n$')
    plt.ylabel(r'$\mathcal{E}_{\mathrm{noise}}$ (noise-induced error)')
    plt.legend()
    utils.ensureDir('plots/')
    plt.savefig('plots/dd_counterexample.pdf', bbox_inches='tight')
    plt.close()


def plot_lower_bound_hypotheses():
    print('Plotting lower bound hypotheses')
    plt.figure(figsize=(5.5, 3))
    plt.grid(True, which='both')
    plt.ylim(0.7, 1.1)
    plt.axvline(30, color='#333333', linestyle='--', linewidth=1)

    sphere_results = load_dd_results('sphere-n30-p30-maxn256-lambda1e-12')
    trained_15_results = load_dd_results('trained-n30-p30-ntrain15-maxn256-lambda1e-12-hidden_256_256')
    trained_60_results = load_dd_results('trained-n30-p30-ntrain60-maxn256-lambda1e-12-hidden_256_256')
    plot_data = []
    plot_data.append((np.array([[dd_gaussian_curve(n, p=30)] for n in range(1, 257)]), '#22AA22',
                      r'$\mathcal{N}(0, \boldsymbol{I}_p)$ (analytic)', dict()))
    plot_data.append((np.array(trained_15_results.results_n), '#AA0000', 'Optimized for $n=15$', dict()))
    plot_data.append((np.array(trained_60_results.results_n), '#FF8800', 'Optimized for $n=60$', dict(linestyle='--')))
    plot_data.append((np.array(sphere_results.results_n), '#0000BB', r'$\mathcal{U}(\mathbb{S}^{p-1})$ (empirical)', dict()))
    plot_data.append((np.array([[dd_sphere_curve(n, p=30) if n <= 30 else 0.0] for n in range(1, 257)]),
                      '#00FFFF', r'\(\mathcal{U}(\mathbb{S}^{p-1}), n \leq p\) (analytic)', dict(linestyle='--')))
    plot_data.append((np.array([[dd_lower_bound(n, p=30)] for n in range(1, 257)]), 'k', 'Lower bound', dict()))

    plt.xscale('log')

    reference_arr = np.mean(np.array(sphere_results.results_n), axis=1)

    for i, ns in enumerate([np.array(list(range(1, 28))), np.array(list(range(33, 257)))]):
        for arr, color, label, kwargs in plot_data:
            arr_selected = arr[ns-1, :]
            means = np.mean(arr_selected, axis=1)
            reference = reference_arr[ns-1]
            num_obs = arr_selected.shape[1]
            stds = np.std(arr_selected, axis=1) / (1.0 if num_obs <= 1 else np.sqrt(num_obs - 1))
            plt.fill_between(ns, (means - stds) / reference, (means + stds) / reference, alpha=0.3,
                             facecolor=color, edgecolor='none')
            if i == 0:  # with label
                plt.plot(ns, means/reference, color=color, label=label, linewidth=1, **kwargs)
            else:
                plt.plot(ns, means/reference, color=color, linewidth=1, **kwargs)

    plt.xlabel('Number of points $n$')
    plt.ylabel('$\mathcal{E}_{\mathrm{noise}}$ relative to \(\mathcal{U}(\mathbb{S}^{p-1})\) (empirical)')
    plt.legend()
    utils.ensureDir('plots/')
    plt.savefig('plots/sphere_optimality.pdf', bbox_inches='tight')
    plt.close()


def plot_lower_bound_hypotheses_2():
    # was only used for checking whether the sphere formula is symmetric, see main.py
    print('Plotting lower bound hypotheses 2')
    plt.figure('Sample-wise double descent', figsize=(5.5, 3))
    plt.grid(True, which='both')
    plt.ylim(0.7, 1.2)
    sphere_results = load_dd_results('sphere-n2-p2-maxn20-lambda1e-12')
    plot_data = []
    p=2
    plot_data.append((np.array([[dd_gaussian_curve(n, p=p)] for n in range(1, 21)]), '#22AA22',
                      r'$\mathcal{N}(0, \boldsymbol{I}_p)$ (analytic)'))
    plot_data.append((np.array(sphere_results.results_n), '#0000BB', r'$\mathcal{U}(\mathbb{S}^{p-1})$ (empirical)'))
    plot_data.append((np.array([[dd_sphere_curve(n, p=p) if n <= p else 0.0] for n in range(1, 21)]),
                      '#00FFFF', r'\(\mathcal{U}(\mathbb{S}^{p-1}), n \leq p\) (analytic)'))
    plot_data.append((np.array([[dd_sphere_curve(n=p, p=n) if n >= p else 0.0] for n in range(1, 21)]),
                      '#00AAAA', r'\(\frac{p}{n-1-p}\cdot\frac{n-2}{n}, n \geq p\)'))
    plot_data.append((np.array([[dd_lower_bound(n, p=p)] for n in range(1, 21)]), 'k', 'Lower bound'))

    plt.xscale('log')

    reference_arr = np.mean(np.array(sphere_results.results_n), axis=1)

    for i, ns in enumerate([np.array(list(range(1, 21)))]):
        for arr, color, label in plot_data:
            arr_selected = arr[ns-1, :]
            means = np.mean(arr_selected, axis=1)
            print(f'{label}: {repr(means)}')
            reference = reference_arr[ns-1]
            num_obs = arr_selected.shape[1]
            stds = np.std(arr_selected, axis=1) / (1.0 if num_obs <= 1 else np.sqrt(num_obs - 1))
            plt.fill_between(ns, (means - stds) / reference, (means + stds) / reference, alpha=0.3, color=color)
            if i == 0:  # with label
                plt.plot(ns, means/reference, color=color, label=label, linewidth=1)
            else:
                plt.plot(ns, means/reference, color=color, linewidth=1)

    plt.axvline(p, color='k', linestyle='--')
    plt.xlabel('Number of points $n$')
    plt.ylabel('$\mathcal{E}_{\mathrm{noise}}$ relative to \(\mathcal{U}(\mathbb{S}^{p-1})\) (empirical)')
    plt.legend(fontsize=8)
    utils.ensureDir('plots/')
    plt.savefig('plots/sphere_optimality_2.pdf', bbox_inches='tight')
    plt.close()


def plot_dd_curves(name, plot_data_emp, plot_data_func, plot_std=True, n=30, p=30, legend_right=True, plot_type='n',
                   figsize=None):
    # generic plotting function that can be reused
    # for example usages see the functions below
    # plot_data_emp should contain tuples (name of the data folder, label, dict with extra plot arguments)
    # plot_data_func should contain tuples (function accepting arguments n and p, label, dict with extra plot arguments)
    plt.figure(figsize=figsize if figsize is not None else ((4.4, 3.5) if legend_right else (5.5, 3.5)))
    plt.grid(True, which='both')
    plt.ylim(2e-2, 1e4)
    plt.axvline(n if plot_type == 'p' else p, color='#333333', linestyle='--', linewidth=1)

    #ns = np.array(list(range(1, 30)) + list(range(31, 257)))  # exclude the critical point
    #lower_bound_arr = np.array([dd_lower_bound(n, d=30) for n in ns])

    if plot_type == 'n':
        plot_data = [(np.array(load_dd_results(name).results_n), label, kwargs)
                     for name, label, kwargs in plot_data_emp]
        plot_data += [(np.array([[func(n, p)] for n in range(1, 257)]), label, kwargs)
                      for func, label, kwargs in plot_data_func]
    elif plot_type == 'n_over':
        plot_data = [(np.array(load_dd_results(name).results_n_over), label, kwargs)
                     for name, label, kwargs in plot_data_emp]
        plot_data += [(np.array([[func(n, p)] for n in range(1, 257)]), label, kwargs)
                      for func, label, kwargs in plot_data_func]
    else:
        plot_data = [(np.array(load_dd_results(name).results_p), label, kwargs)
                     for name, label, kwargs in plot_data_emp]
        plot_data += [(np.array([[func(n, p)] for p in range(1, 257)]), label, kwargs)
                      for func, label, kwargs in plot_data_func]

    plt.yscale('log')
    plt.xscale('log')

    ns = np.array(list(range(1, 257)))  # assume n runs from 1 to 256
    for arr, label, kwargs in plot_data:
        arr_selected = arr[ns-1, :]
        means = np.mean(arr_selected, axis=1)
        if plot_std:
            stds = np.std(arr_selected, axis=1)
            if arr_selected.shape[1] >= 2:
                stds /= np.sqrt(arr_selected.shape[1]-1)
            plt.fill_between(ns, np.clip(means - stds, 1e-5, 1e15), means + stds, alpha=0.3,
                             facecolor=kwargs.get('color', 'k'), edgecolor='none')

        plt.plot(ns, means, label=label, linewidth=1, **kwargs)

    plt.xlabel('Number of parameters $p$' if plot_type == 'p' else 'Number of points $n$')
    plt.ylabel(r'$\mathcal{E}_{\mathrm{noise}}$ (noise-induced error)')
    if legend_right:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc='best')
    utils.ensureDir('plots/')
    plt.savefig(f'plots/{name}.pdf', bbox_inches='tight')
    plt.close()


def plot_ntk():
    print('Plotting NTK results')
    for plot_type in ['n', 'n_over']:
        plot_data_emp = [(f'ntk-{act.name}-n30-maxn256-layers_4_6_1-lambda_1e-12',
                          f'{act.name} NTK', act.kwargs) for act in get_activation_functions()]
        plot_data_emp.append(('sphere-n30-p30-maxn256-lambda1e-12', r'\(\mathcal{U}(\mathbb{S}^{p-1})\)', dict(color='#0000BB')))
        plot_data_func = [(dd_lower_bound, r'Lower bound', dict(color='k'))]
        plot_dd_curves(f'ntk_results_{plot_type}', plot_data_emp, plot_data_func, plot_std=True, plot_type=plot_type)


def plot_nn():
    print('Plotting NN Results')
    for plot_type in ['n', 'n_over', 'p']:
        plot_data_emp = [(f'nn-{act.name}-n30-maxn256-p30-layers_10_256_256_256-lambda_1e-12',
                          f'{act.name} NN', act.kwargs) for act in get_activation_functions()]
        if plot_type != 'p':
            plot_data_emp.append(('sphere-n30-p30-maxn256-lambda1e-12', r'\(\mathcal{U}(\mathbb{S}^{p-1})\)', dict(color='#0000BB')))
        plot_data_func = [(dd_lower_bound, r'Lower bound', dict(color='k'))]
        plot_dd_curves(f'nn_results_{plot_type}', plot_data_emp, plot_data_func, plot_std=True, plot_type=plot_type)


def plot_rff():
    print('Plotting RFF Results for small weight gain')
    for plot_type in ['n', 'n_over']:
        plot_data_emp = [('rffbias-n30-p30-maxn256-d10-lambda1e-12-wg0.03333333333333333', 'RFF (cos with bias)', dict(color='#FF8800')),
                         ('rffsincos-n30-p30-maxn256-d10-lambda1e-12-wg0.03333333333333333', 'RFF (sin and cos)', dict(color='#00AA00')),
                         ('sphere-n30-p30-maxn256-lambda1e-12', r'\(\mathcal{U}(\mathbb{S}^{p-1})\)', dict(color='#0000BB'))]
        plot_data_func = [(dd_lower_bound, r'Lower bound', dict(color='k'))]
        plot_dd_curves(f'rff_results_{plot_type}', plot_data_emp, plot_data_func, plot_std=True, plot_type=plot_type)


def plot_rff_vs_sphere():
    print('Plotting RFF relative to sphere')
    plt.figure(figsize=(5.5, 3))
    plt.grid(True, which='both')
    plt.ylim(0.7, 1.1)
    plt.axvline(30, color='#333333', linestyle='--', linewidth=1)

    sphere_results = load_dd_results('sphere-n30-p30-maxn256-lambda1e-12')
    rffbias_results = load_dd_results('rffbias-n30-p30-maxn256-d10-lambda1e-12-wg1.0')
    rffsincos_results = load_dd_results('rffsincos-n30-p30-maxn256-d10-lambda1e-12-wg1.0')
    plot_data = []
    plot_data.append((np.array([[dd_gaussian_curve(n, p=30)] for n in range(1, 257)]), '#22AA22',
                      r'$\mathcal{N}(0, \boldsymbol{I}_p)$ (analytic)', dict()))
    plot_data.append((np.array(sphere_results.results_n), '#0000BB', r'$\mathcal{U}(\mathbb{S}^{p-1})$ (empirical)', dict()))
    plot_data.append((np.array([[dd_sphere_curve(n, p=30) if n <= 30 else 0.0] for n in range(1, 257)]),
                      '#00FFFF', r'\(\mathcal{U}(\mathbb{S}^{p-1}), n \leq p\) (analytic)', dict(linestyle='--')))
    plot_data.append((np.array(rffbias_results.results_n), '#AA0000', 'RFF (cos with bias)', dict()))
    plot_data.append((np.array(rffsincos_results.results_n), '#FF8800', 'RFF (sin and cos)', dict(linestyle='--')))
    plot_data.append((np.array([[dd_lower_bound(n, p=30)] for n in range(1, 257)]), 'k', 'Lower bound', dict()))

    plt.xscale('log')

    reference_arr = np.mean(np.array(sphere_results.results_n), axis=1)

    for i, ns in enumerate([np.array(list(range(1, 28))), np.array(list(range(33, 257)))]):
        for arr, color, label, kwargs in plot_data:
            arr_selected = arr[ns-1, :]
            means = np.mean(arr_selected, axis=1)
            reference = reference_arr[ns-1]
            num_obs = arr_selected.shape[1]
            stds = np.std(arr_selected, axis=1) / (1.0 if num_obs <= 1 else np.sqrt(num_obs - 1))
            plt.fill_between(ns, (means - stds) / reference, (means + stds) / reference, alpha=0.3,
                             facecolor=color, edgecolor='none')
            if i == 0:  # with label
                plt.plot(ns, means/reference, color=color, label=label, linewidth=1, **kwargs)
            else:
                plt.plot(ns, means/reference, color=color, linewidth=1, **kwargs)

    plt.xlabel('Number of points $n$')
    plt.ylabel('$\mathcal{E}_{\mathrm{noise}}$ relative to \(\mathcal{U}(\mathbb{S}^{p-1})\) (empirical)')
    plt.legend()
    utils.ensureDir('plots/')
    plt.savefig('plots/rff_vs_sphere.pdf', bbox_inches='tight')
    plt.close()


def plot_poly_kernel():
    print('Plotting poly kernel results')
    plot_data_emp = [('poly-deg4-inputdim3-c1-maxn256-lambda1e-12', r'Polynomial kernel', dict(color='#1f77b4'))]
    plot_data_func = [(dd_lower_bound, r'Lower bound', dict(color='k'))]
    plot_dd_curves('poly_results_n', plot_data_emp, plot_data_func, p=35, legend_right=False, plot_std=True)

