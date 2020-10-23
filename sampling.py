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

import torch
import numpy as np
import math
import torch.nn as nn


def get_default_device():
    # return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return 'cpu'  # CPU is much faster for SVD than GPU


def poly_kernel_fm_rec(x, deg, idx, factor):
    # recursive helper function for the feature map of a polynomial kernel
    # idx is the dimension-index indicating that powers of x[idx] should be added next
    if idx+1 == x.shape[1]:
        return np.sqrt(factor // math.factorial(deg)) * (x[:, -1:] ** deg)

    features = []
    for m in range(deg+1):
        new_features = x[:, idx:idx+1]**m * poly_kernel_fm_rec(x, deg-m, idx+1, factor // math.factorial(m))
        features.append(new_features)
    return torch.cat(features, dim=1)


def poly_kernel_fm(x, deg, c):
    # compute the result of a feature map for the polynomial kernel with degree deg and additive constant c
    # x should be of shape n_parallel x n_samples x dim
    x = torch.cat([x, np.sqrt(c) * torch.ones_like(x[:, 0:1])], dim=1)
    factor = math.factorial(deg)
    return poly_kernel_fm_rec(x, deg, 0, factor)

# ----- Samplers that compute (batched) samples from a certain distribution


class NormalXSampler:  # samples from a standard normal distribution
    def __init__(self, dim):
        self.dim = dim

    def sample(self, *sizes):
        return torch.randn(*sizes, self.dim, dtype=torch.float64, device=get_default_device())


class SphereXSampler:  # samples from a uniform distribution on a sphere
    def __init__(self, dim):
        # Sample from \bbS^{dim-1}
        self.sampler = NormalXSampler(dim)
        self.dim = dim

    def sample(self, *sizes):
        x = self.sampler.sample(*sizes)
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class UnitXSampler:  # unused, samples from a uniform distribution on a unit interval
    def __init__(self):
        self.dim = 1

    def sample(self, *sizes):
        result = torch.zeros(*sizes, self.dim, dtype=torch.float64, device=get_default_device())
        result.uniform_(-1.0, 1.0)
        return result


class WeightActLayer(nn.Module):  # linear layer with activation, custom initialization
    def __init__(self, d_in, d_out, act, weight_factor, use_bias=False):
        super().__init__()
        self.act = act
        self.weight = nn.Parameter(torch.zeros(d_out, d_in, dtype=torch.float64))
        with torch.no_grad():
            self.weight.normal_(0.0, 1.0)
        self.weight_factor = weight_factor / (np.sqrt(d_in))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(d_out, dtype=torch.float64))

    def forward(self, x):
        x = self.weight_factor * (x.matmul(self.weight.t()))
        if self.use_bias:
            x = x + self.bias[None, :]
        return self.act(x)


class ParallelWeightActLayer(nn.Module):
    # like WeightActLayer, but with an extra dimension such that multiple NNs can be represented in this extra dimension
    def __init__(self, n_parallel, d_in, d_out, act, weight_factor, use_bias=False, device='cpu'):
        super().__init__()
        self.act = act
        self.weight = nn.Parameter(torch.zeros(n_parallel, d_out, d_in, dtype=torch.float64, device=device))
        with torch.no_grad():
            self.weight.normal_(0.0, 1.0)
        self.use_bias = use_bias
        if self.use_bias:
            self.weight_factor = weight_factor / (np.sqrt(d_in))
            self.bias = nn.Parameter(torch.zeros(n_parallel, d_out, dtype=torch.float64, device=device))
        else:
            self.weight_factor = weight_factor / (np.sqrt(d_in))

    def forward(self, x):
        x = self.weight_factor * (x.bmm(self.weight.transpose(1, 2)))
        if self.use_bias:
            x = x + self.bias[:, None, :]
        return self.act(x)


class ActLayer(nn.Module):
    # layer that only represents an activation function
    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


class RandomNNSampler:
    # sampler that takes an x distribution via an x_sampler and then samples z via a random NN feature map
    def __init__(self, act, x_sampler, hidden_sizes, d_out):
        self.act = act
        self.x_sampler = x_sampler
        self.d_in = x_sampler.dim
        self.d_out = d_out
        self.hidden_sizes = hidden_sizes
        self.device = get_default_device()
        self.dim = self.d_out

    def sample(self, N, n):
        # sample a N x n x self.dim tensor, where self.dim corresponds to p
        # and N corresponds to the number of feature maps used in parallel
        layer_sizes = [self.d_in] + self.hidden_sizes + [self.d_out]
        weight_factor = 1.0/self.act(torch.randn(10000, dtype=torch.float64, device=self.device)).std().item()
        weight_factors = [1.0] + [weight_factor]*len(self.hidden_sizes)
        model = nn.Sequential(*[ParallelWeightActLayer(N, d_in, d_out, self.act, weight_factor, use_bias=False)
                                for (d_in, d_out, weight_factor)
                                in zip(layer_sizes[:-1], layer_sizes[1:], weight_factors)])
        x_in = self.x_sampler.sample(N, n)
        with torch.no_grad():
            return model(x_in)


def set_in_array(arr, idx, val):
    # helper function used in a lambda below
    arr[idx] = val


def identity(x):  # can be used as a non-activation function
    return x


class RandomNTKSampler:
    # samples from a NTK feature map corresponding to a random neural network
    # the random NTK sampler is implemented by grabbing intermediate tensors from the forward and backward pass
    # and using these to compute the derivatives for individual samples
    # since otherwise, backward() would need to be called for each sample individually and that might be quite slow
    # due to less parallelization
    def __init__(self, act, x_sampler, hidden_sizes, d_out):
        self.act = act
        self.x_sampler = x_sampler
        self.d_in = x_sampler.dim
        self.d_out = d_out
        self.hidden_sizes = hidden_sizes
        self.device = get_default_device()
        self.layer_sizes = [self.d_in] + self.hidden_sizes + [self.d_out]
        self.dim = sum([lsa * lsb for lsa, lsb in zip(self.layer_sizes[:-1], self.layer_sizes[1:])])

    def sample(self, N, n):
        # sample a N x n x self.dim tensor, where self.dim corresponds to p
        # and N corresponds to the number of feature maps used in parallel
        weight_factor = 1.0/self.act(torch.randn(10000, dtype=torch.float64, device=self.device)).std().item()
        weight_factors = [1.0] + [weight_factor]*len(self.hidden_sizes)
        # cannot put self.act as activation in the weight layers
        # since we need to grab the grad_output wrt to the tensor before the act
        # therefore, add self.act in an extra layer later
        weight_layers = [ParallelWeightActLayer(N, d_in, d_out, identity, weight_factor)
                                for (d_in, d_out, weight_factor)
                                in zip(self.layer_sizes[:-1], self.layer_sizes[1:], weight_factors)]
        acts = [self.act] * (len(weight_layers) - 1) + [identity]  # do not use an activation after the last layer
        model = nn.Sequential(*[nn.Sequential(wl, ActLayer(act_fn)) for wl, act_fn in zip(weight_layers, acts)])
        wl_inputs = [None] * len(weight_layers)
        wl_grad_outputs = [None] * len(weight_layers)
        for i in range(len(weight_layers)):
            weight_layers[i].register_forward_hook(
                lambda wl, inp, output, i=i: set_in_array(wl_inputs, i, inp[0].detach().clone()))
            weight_layers[i].register_backward_hook(
                lambda wl, grad_input, grad_output, i=i: set_in_array(wl_grad_outputs, i,
                                                                      grad_output[0].detach().clone()))
        model = model.to(self.device)
        x_in = self.x_sampler.sample(N, n)

        out = model(x_in)
        out.backward(torch.ones_like(out))

        result = torch.cat([(weight_layers[i].weight_factor * wl_inputs[i][:, :, None, :]
                             * wl_grad_outputs[i][:, :, :, None]).view(N, n, -1)
                          for i in range(len(weight_layers))], dim=2)

        if n == 1:
            # can check if NTK computation is correct via gradient attributes
            grad_result = torch.cat([weight_layers[i].weight.grad.reshape(N, n, -1)
                            for i in range(len(weight_layers))], dim=2)
            assert(torch.allclose(result, grad_result))

        return result


class RFFBiasSampler:
    def __init__(self, x_sampler, d_out, weight_gain=1.0):
        self.x_sampler = x_sampler
        self.dim = d_out
        self.weight_gain = weight_gain

    def sample(self, N, n):
        # sample a N x n x self.dim tensor, where self.dim corresponds to p
        # and N corresponds to the number of feature maps used in parallel
        x = self.x_sampler.sample(N, n)  # N x n x d
        W = self.weight_gain * torch.randn(N, self.x_sampler.dim, self.dim, dtype=torch.float64)
        b = 2 * np.pi * torch.rand(N, 1, self.dim, dtype=torch.float64)
        return torch.cos(x.bmm(W) + b)


class RFFSinCosSampler:
    def __init__(self, x_sampler, d_out, weight_gain=1.0):
        self.x_sampler = x_sampler
        assert d_out % 2 == 0, 'Output dimension must be a multiple of 2 (use sin/cos pairs)'
        self.dim = d_out
        self.weight_gain = weight_gain

    def sample(self, N, n):
        # sample a N x n x self.dim tensor, where self.dim corresponds to p
        # and N corresponds to the number of feature maps used in parallel
        x = self.x_sampler.sample(N, n)  # N x n x d
        W = self.weight_gain * torch.randn(N, self.x_sampler.dim, self.dim//2, dtype=torch.float64)
        x = x.bmm(W)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=2)


class FixedFeatureMapSampler:
    # takes a fixed (non-random) feature map and an x sampler and provides a sampler for z = featuremap(x)
    def __init__(self, x_sampler, model, dim, no_grad=False):
        # model is the feature map, dim the output dimension of the feature map, x_sampler the input sampler
        self.x_sampler = x_sampler
        self.model = model
        self.dim = dim
        self.no_grad = no_grad

    def sample(self, *sizes):
        x = self.x_sampler.sample(*sizes)
        if self.no_grad:
            with torch.no_grad():
                return self.model(x.view(-1, self.x_sampler.dim)).view(*sizes, self.dim)
        else:
            return self.model(x.view(-1, self.x_sampler.dim)).view(*sizes, self.dim)

