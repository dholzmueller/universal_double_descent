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

from sampling import *
import torch.nn.functional as F
import utils
from pathlib import Path


class DoubleDescentResults:
    # This class is used to compute and store Monte Carlo estimates of Enoise.
    # Objects of this class can then be serialized and saved on disk.
    def __init__(self, n, max_n, p, max_p, n_parallel, n_mc=10000, random_seed=0, lam=1e-12, compute_p_results=True):
        # n_parallel individual estimates of Enoise will be computed for each combination (n, p)
        # n_mc is the number of Monte Carlo points used to estimate \Sigma in each individual estimate
        # lam is the regularization parameter \lambda
        # Enoise is computed for p=self.p and n in range(1, self.max_n+1)
        # If compute_p_results is True, then Enoise is also computed for n=self.n and p in range(1, self.max_p+1)
        self.n = n
        self.max_n = max_n
        self.p = p
        self.max_p = max_p
        self.n_parallel = n_parallel
        self.n_mc = n_mc
        self.random_seed = random_seed
        self.lam = lam  # regularization
        self.ns = list(range(1, max_n+1))
        self.results_n = np.zeros(shape=(max_n, n_parallel))
        self.results_n_over = np.zeros(shape=(max_n, n_parallel))
        self.compute_p_results = compute_p_results
        if self.compute_p_results:
            self.ps = list(range(1, max_p + 1))
            self.results_p = np.zeros(shape=(max_p, n_parallel))
        else:
            self.ps = [p]
            self.results_p = np.zeros(shape=(1, n_parallel))

    def compute_trace(self, Z, Sigma_p):
        # Helper function, computes the desired value tr((Z^+)^T \Sigma Z^+)
        try:
            U, S, V = Z.svd()
            diag_reg = S / (S ** 2 + self.lam)
            trace = torch.einsum('bji,bjk,bki,bi->b', V, Sigma_p, V, diag_reg ** 2)
        except:
            print('SVD failed, resorting to classical formula for (regularized) pseudoinverse')
            if Z.shape[1] < Z.shape[2]:  # overparameterized case
                X_pinv = Z.transpose(1, 2).bmm(
                    (Z.bmm(Z.transpose(1, 2)) + self.lam * torch.eye(Z.shape[1])[None, :, :]).inverse())
            else:  # underparameterized case
                X_pinv = (Z.transpose(1, 2).bmm(Z) + self.lam * torch.eye(Z.shape[2])[None, :, :]).inverse().bmm(
                    Z.transpose(1, 2))
            prod = X_pinv.bmm(X_pinv.transpose(1, 2))
            trace = (Sigma_p * prod).sum(dim=2).sum(dim=1)
        return trace

    def compute(self, sampler):
        # computes Enoise estimates for the given sampler that samples the z values
        torch.manual_seed(self.random_seed)

        # compute the results in batches to keep RAM usage moderate
        max_parallel_batch_size = 100
        Z_data = []
        Sigma = []
        parallel_batch_sizes = [max_parallel_batch_size] * (self.n_parallel//max_parallel_batch_size)
        remainder = self.n_parallel % max_parallel_batch_size
        if remainder > 0:
            parallel_batch_sizes.append(remainder)

        for pbs in parallel_batch_sizes:
            print('.', end='', flush=True)
            Z = sampler.sample(pbs, self.max_n + self.n_mc)
            Z_mc = Z[:, :self.n_mc, :]
            Z_data.append(Z[:, self.n_mc:, :].clone())
            Sigma.append(Z_mc.transpose(1, 2).bmm(Z_mc) / self.n_mc)
        print()
        Z_data = torch.cat(Z_data, dim=0)
        Sigma = torch.cat(Sigma, dim=0)

        # compute results_n, i.e. results for fixed p and varying n
        Sigma_p = Sigma[:, :self.p, :self.p]
        vals, vecs = Sigma_p.symeig(eigenvectors=True)
        Sigma_p_inv = torch.einsum('bij,bj,bkj->bik', vecs, vals/(vals**2+self.lam), vecs)
        #Sigma_p_inv = Sigma_p.pinverse()
        for n in range(1, self.max_n+1):
            print('.', end='', flush=True)
            Z_n = Z_data[:, :n, :self.p]
            trace = self.compute_trace(Z_n, Sigma_p)
            self.results_n[n-1, :] = trace.detach().cpu().numpy()
            if n < self.p:
                WWT = Z_n.bmm(Sigma_p_inv.bmm(Z_n.transpose(1, 2)))
                eigvals, _ = WWT.symeig()
                self.results_n_over[n-1, :] = (eigvals/(eigvals**2+self.lam)).sum(dim=1).detach().cpu().numpy()
            else:
                self.results_n_over[n-1, :] = self.results_n[n-1, :]

        print()

        if self.compute_p_results:
            # compute results_p, i.e. results for fixed n and varying p
            for p in range(1, self.max_p + 1):
                print('.', end='', flush=True)
                Z_d = Z_data[:, :self.n, :p]
                trace = self.compute_trace(Z_d, Sigma[:, :p, :p])
                self.results_p[p - 1, :] = trace.detach().cpu().numpy()
            print()
        else:
            self.results_p[0, :] = self.results_n[self.n - 1, :]


def train_best_feature_map(name, layer_sizes, n, act, n_iterations=1000, n_mc=1000, batch_size=1024, last_layer_act=True):
    # Trains a feature map to minimize Enoise for the given value of n
    torch.manual_seed(0)
    device = get_default_device()
    weight_factor = act(torch.randn(10000, dtype=torch.float64, device=device)).std().item()
    weight_factors = [1.0] + [weight_factor] * len(layer_sizes[1:-1])
    x_sampler = NormalXSampler(dim=layer_sizes[0])
    acts = [act] * (len(layer_sizes) - 2) + [act if last_layer_act else identity]
    model = nn.Sequential(*[WeightActLayer(d_in, d_out, act_fn, weight_factor, use_bias=True)
                            for (d_in, d_out, weight_factor, act_fn)
                            in zip(layer_sizes[:-1], layer_sizes[1:], weight_factors, acts)])
    filename = Path('models')/name/'model.p'
    if utils.existsFile(filename):
        print('Loading serialized model')
        model.load_state_dict(utils.deserialize(filename))
        model = model.to(device)
    else:
        model = model.to(device)
        max_lr = 1e-3
        opt = torch.optim.Adam(model.parameters(), lr=max_lr, betas=(0.9, 0.999), amsgrad=True)
        lam = 1e-12
        for i in range(n_iterations):
            print(f'Iteration {i+1}/{n_iterations}')
            for group in opt.param_groups:
                group['lr'] = max_lr*(1-i/n_iterations)
            x_cov = x_sampler.sample(n_mc)
            z_cov = model(x_cov)
            Sigma = z_cov.t().matmul(z_cov) / n_mc
            Z = model(x_sampler.sample(n * batch_size)).view(batch_size, n, layer_sizes[-1])
            if n < layer_sizes[-1]:  # overparameterized case
                X_pinv = Z.transpose(1, 2).bmm((Z.bmm(Z.transpose(1, 2)) + lam*torch.eye(Z.shape[1])[None, :, :]).inverse())
            else:  # underparameterized case
                X_pinv = (Z.transpose(1, 2).bmm(Z) + lam*torch.eye(Z.shape[2])[None, :, :]).inverse().bmm(Z.transpose(1, 2))
            prod = X_pinv.bmm(X_pinv.transpose(1, 2))
            mean_trace = (Sigma[None, :, :] * prod).sum(dim=2).sum(dim=1).mean()
            print('Mean trace:', mean_trace.item())
            mean_trace.backward()
            opt.step()
            opt.zero_grad()
        utils.serialize(filename, model.state_dict())
    return FixedFeatureMapSampler(x_sampler, model, dim=layer_sizes[-1], no_grad=True)


def compute_dd_results(name, sampler, n_rep=10, n_parallel=1000, **kwargs):
    # computes the results in multiple repetitions and saves them
    # but only if the results are not already computed
    for rep in range(n_rep):
        print(f'Repetition {rep+1}/{n_rep}')
        filename = Path('data/double_descent/') / name / f'v{rep}_{n_parallel}.p'
        if utils.existsFile(filename):
            print('Results have already been computed')
            continue
        results = DoubleDescentResults(**kwargs, random_seed=rep, n_parallel=n_parallel)
        results.compute(sampler)
        utils.serialize(filename, results)


def compute_random_nn_results(layer_sizes, n=30, max_n=256, p=30, n_parallel=1000, n_rep=10, lam=1e-12, act_names=None):
    # n_layers is the number of layers (n_layers-1 is the number of hidden layers)
    # if act_names is not None, then only the activation functions in act_names will be plotted
    for act in get_activation_functions():
        if act_names is not None and act.name not in act_names:
            continue
        print(f'Compute results for activation {act.name}')
        layer_string = '_'.join([str(sz) for sz in layer_sizes])
        name = f'nn-{act.name}-n{n}-maxn{max_n}-p{p}-layers_{layer_string}-lambda_{lam:g}'
        sampler = RandomNNSampler(act=act.func, x_sampler=NormalXSampler(dim=layer_sizes[0]), hidden_sizes=layer_sizes[1:-1],
                                  d_out=layer_sizes[-1])
        compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam)


def compute_random_ntk_results(layer_sizes, n=30, max_n=256, n_parallel=1000, n_rep=10, lam=1e-12):
    # n_layers is the number of layers (n_layers-1 is the number of hidden layers)
    for act in get_activation_functions():
        print(f'Compute NTK results for activation {act.name}')
        layer_string = '_'.join([str(sz) for sz in layer_sizes])
        name = f'ntk-{act.name}-n{n}-maxn{max_n}-layers_{layer_string}-lambda_{lam:g}'
        sampler = RandomNTKSampler(act=act.func, x_sampler=NormalXSampler(dim=layer_sizes[0]), hidden_sizes=layer_sizes[1:-1],
                                  d_out=layer_sizes[-1])
        p = sum([lsa*lsb for lsa, lsb in zip(layer_sizes[:-1], layer_sizes[1:])])
        compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=p, lam=lam,
                           compute_p_results=False)


def compute_best_nn_results(n=30, max_n=256, p=30, n_train=15, n_parallel=1000, n_rep=10, lam=1e-12, hidden_sizes=[256, 256]):
    print('Compute best NN results')
    hidden_string = '_'.join([str(sz) for sz in hidden_sizes])
    name = f'trained-n{n}-p{p}-ntrain{n_train}-maxn{max_n}-lambda{lam:g}-hidden_{hidden_string}'
    sampler = train_best_feature_map(name, layer_sizes=[p] + hidden_sizes + [p], n=n_train, act=torch.tanh, n_iterations=1000)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam, 
                       compute_p_results=False)


def compute_sphere_results(n=30, max_n=256, p=30, n_parallel=1000, n_rep=10, lam=1e-12):
    print('Compute sphere results')
    name = f'sphere-n{n}-p{p}-maxn{max_n}-lambda{lam:g}'
    sampler = SphereXSampler(dim=p)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam,
                       compute_p_results=False)


def compute_normal_results(n=30, max_n=256, p=30, max_p=256, n_parallel=1000, n_rep=10, lam=1e-12):
    print('Compute normal results')
    name = f'normal-n{n}-p{p}-maxn{max_n}-maxp{max_p}-lambda{lam:g}'
    sampler = NormalXSampler(dim=max_p)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam)


def compute_poly_kernel_results(max_n=256, deg=4, input_dim=3, c=1.0, n_parallel=1000, n_rep=10, lam=1e-12):
    print('Compute poly kernel results')
    name = f'poly-deg{deg}-inputdim{input_dim}-c{c:g}-maxn{max_n}-lambda{lam:g}'
    sampler = NormalXSampler(dim=input_dim)
    p = math.factorial(deg + input_dim)//(math.factorial(deg) * math.factorial(input_dim))  # binomial coefficient
    sampler = FixedFeatureMapSampler(sampler, lambda x: poly_kernel_fm(x, deg, c), dim=p, no_grad=True)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=p, max_n=max_n, p=p, max_p=sampler.dim, lam=lam,
                                   compute_p_results=False)


def compute_rff_results(n=30, max_n=256, p=30, d=10, n_parallel=1000, n_rep=10, lam=1e-12, weight_gain=1.0):
    print('Compute RFF bias results')
    name = f'rffbias-n{n}-p{p}-maxn{max_n}-d{d}-lambda{lam:g}-wg{weight_gain}'
    sampler = RFFBiasSampler(NormalXSampler(dim=d), d_out=p, weight_gain=weight_gain)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam,
                       compute_p_results=False)

    print('Compute RFF sin+cos results')
    name = f'rffsincos-n{n}-p{p}-maxn{max_n}-d{d}-lambda{lam:g}-wg{weight_gain}'
    sampler = RFFSinCosSampler(NormalXSampler(dim=d), d_out=p, weight_gain=weight_gain)
    compute_dd_results(name, sampler, n_rep, n_parallel, n=n, max_n=max_n, p=p, max_p=sampler.dim, lam=lam,
                       compute_p_results=False)



def verify_frk(sampler, oversampling_factor=3, n_mc=10000):
    torch.manual_seed(0)
    Z = sampler.sample(n_mc, oversampling_factor*sampler.dim)
    _, singvals, _ = Z.svd()
    print('Verifying (FRK) for analytic feature map:')
    # use 1e-30 to protect against division by zero
    print(f'Maximum observed inverse condition number for {oversampling_factor}x oversampling:',
          f'{(singvals[:, -1]/(singvals[:, 0] + 1e-30)).max().item():g}')


def verify_frk_for_ntk(layer_sizes):
    acts = [act for act in get_activation_functions() if act.is_analytic]
    for act in acts:
        print(f'Checking activation function {act.name}')
        sampler = RandomNTKSampler(act=act.func, x_sampler=NormalXSampler(dim=layer_sizes[0]),
                                   hidden_sizes=layer_sizes[1:-1],
                                   d_out=layer_sizes[-1])
        verify_frk(sampler)


class ActivationFunction:
    # represents an activation function and some information about it, including plotting information
    def __init__(self, name, func, is_analytic, **kwargs):
        self.name = name
        self.func = func
        self.is_analytic = is_analytic
        self.kwargs = kwargs


def get_activation_functions():
    act_funs = [
                ActivationFunction('sigmoid', torch.sigmoid, is_analytic=True, color='#CC6600'),  # orange
                ActivationFunction('tanh', torch.tanh, is_analytic=True, color='#CCCC00'),  # yellow
                ActivationFunction('GELU', F.gelu, is_analytic=True, color='#AA0000'),  # red
                ActivationFunction('softplus', F.softplus, is_analytic=True, color='#6666FF'),  # light blue
                ActivationFunction('ReLU', torch.relu, is_analytic=False, color='#00FFFF', linestyle='--'),  # cyan
                ActivationFunction('SELU', torch.selu, is_analytic=False, color='#AA00AA', linestyle='--'),  # magenta
                ActivationFunction('ELU', F.elu, is_analytic=False, color='#008888', linestyle='--')  # dark cyan
               ]
    return act_funs


def dd_lower_bound(n, p):
    if p < n:
        return p / (n + 1 - p)
    else:
        return n / (p + 1 - n)


def dd_sphere_curve(n, p):
    # return the analytic values for the sphere
    if n == 1:
        return 1/p
    elif p == 1:
        return 1/n
    elif n+1 < p:
        return n * (p - 2) / (p * (p - n - 1))
    elif p < n:
        return np.nan  # unknown
    else:
        return np.inf


def dd_gaussian_curve(n, p):
    # return the analytic values for the standard Gaussian distribution
    if n+1 < p:
        return n / (p - n - 1)
    elif p+1 < n:
        return p / (n - p - 1)
    else:
        return np.inf


def test_poly_kernel():
    deg = 4
    input_dim = 3
    c = 1.0
    z = NormalXSampler(dim=input_dim).sample(10)
    x = poly_kernel_fm(z, deg=deg, c=c)

    # dimension p of the feature map should correspond to the computed dimension
    assert(x.shape[2] == 35)

    # check that the feature map phi satisfies phi(x)^T phi(y) = (x^T y + c)^{deg}
    assert(torch.allclose((torch.einsum('ijk,ijk->ij', z, z)+c)**deg, torch.einsum('ijk,ijk->ij', x, x)))


def test_ntk():
    ntk_sampler = RandomNTKSampler(act=torch.tanh, x_sampler=NormalXSampler(dim=3), hidden_sizes=[6], d_out=1)
    ntk_sampler.sample(15, 1)  # use 1 in the second argument since this triggers the assertion

