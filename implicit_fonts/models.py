import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

custom = False
verbose = False

ratio = 1

class ReLULayer_custom(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.omega_0 > 1:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / (self.omega_0 * self.in_features),
                                        1 / (self.omega_0 * self.in_features))  # +1 + 1 / self.in_features

    def forward(self, input):
        return torch.nn.functional.leaky_relu(self.linear(input), negative_slope=0.1)


class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.omega_0 > 1:
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / (self.omega_0 * self.in_features),
                                        1 / (self.omega_0 * self.in_features))  # +1 + 1 / self.in_features

    def forward(self, input):
        return torch.nn.functional.leaky_relu(self.linear(input), negative_slope=0.1)


class ReLUNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(ReLULayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))
        # self.net.append(LayerNorm([1,1,hidden_features]))
        for i in range(hidden_layers):
            if i == hidden_layers//2:
                self.net.append(ReLULayer(hidden_features+in_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
            else:
                self.net.append(ReLULayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))
            # self.net.append(LayerNorm([1,1,hidden_features]))
        self.omega = first_omega_0

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            self.net.append(final_linear)

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        coords, coords_fused = input
        #         output = self.net(coords)
        output = coords_fused  # self.net(coords)
        skip_idx = 1 + (len(self.net)-2)//2
        for i in range(len(self.net) - 1):
            if i == skip_idx:
                output = torch.cat([output, coords_fused], dim=-1)
            output = self.net[i](output * self.omega)
        output = self.net[i + 1](output)
        return output, coords


class AutoDecoderFamily(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., num_embed=2, num_glyphs=26):
        super().__init__()
        embed_features = 32
        self.embed = nn.Embedding(num_embed, embed_features)
        self.omega = first_omega_0
        # self.init_embed_weight()
        # self.embed_net = []
        # self.embed_net.append(ReLULayer(embed_features, embed_features, omega_0=first_omega_0))
        # self.embed_net.append(ReLULayer(embed_features, embed_features, omega_0=first_omega_0))
        # self.embed_net.append(ReLULayer(embed_features, embed_features, omega_0=first_omega_0))

        # self.embed_net = nn.Sequential(*self.embed_net)
        self.network = ReLUNetwork(in_features=in_features+embed_features+num_glyphs, out_features=out_features, hidden_features=hidden_features,
                  hidden_layers=hidden_layers, outermost_linear=outermost_linear, first_omega_0=first_omega_0, hidden_omega_0=hidden_omega_0)

    def init_embed_weight(self):
        weights = torch.ones_like(self.embed.weight, requires_grad=True).to(self.embed.weight.device)*0.5
        self.embed.weight = torch.nn.Parameter(weights)

    def forward(self, batch):
        idx, glyph_idx, input = batch
        bs, wh, c = input.shape
        input = input.clone().requires_grad_(True)

        label_embed = self.embed(idx)#
        label_embed_repeat = label_embed[:, None, :].repeat([1,wh,1])
        glyph_idx = glyph_idx[:, None, :].repeat([1,wh,1])
        x = torch.cat([input, glyph_idx, label_embed_repeat], dim=-1)
        output = self.network([input, x])
        return output, label_embed

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
