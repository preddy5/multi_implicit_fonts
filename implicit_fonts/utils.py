
import numpy as np
import skimage
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
import math

import kornia
dsample = kornia.transform.PyrDown()
import time
import torch.nn.functional as F

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    pix_distance = 1/sidelen
    tensors = tuple(dim * [torch.linspace(-1+pix_distance, 1-pix_distance, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    # mgrid = mgrid.reshape(-1, dim)
    return mgrid


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    return X[:,:,:3]

def antialias_kernel(r):
    r = -r
    output = (0.5 + 0.25 * (torch.pow(r, 3) - 3 * r))
    #   output = -0.5*r + 0.5
    return output


def render_sdf(sdf, resolution):
    normalization = resolution  # 0.70710678*2/resolution
    normalized_sdf = sdf / normalization
    clamped_sdf = torch.clamp(normalized_sdf, min=-1, max=1)
    opacity = antialias_kernel(clamped_sdf)  # multiply by color here
    return opacity


def select_top2(tensor, return_indices=False):
    median = torch.median(tensor, dim=-1, keepdims=True)[0]
    diff = torch.abs((tensor-median))
    val, indices = torch.topk(diff, 2, largest=False)
    t= torch.gather(tensor, dim=-1, index=indices)
    if return_indices:
        return t, indices
    return t


def perm_inv_mse(x, y):
    first = (x[:, :, :1] - y) ** 2
    first_sort, first_indices = torch.sort(first, dim=-1, descending=False)

    y_second = torch.gather(y, dim=-1, index=first_indices[:, :, 1:])
    second = (x[:, :, 1:2] - y_second) ** 2
    second_sort, second_indices = torch.sort(second, dim=-1, descending=False)

    y_third = torch.gather(y_second, dim=-1, index=second_indices[:, :, 1:])
    third = (x[:, :, 2:] - y_third) ** 2
    return first_sort[:, :, :1] + second_sort[:, :, :1] + third


def perm_inv_mse_channel(x, y):
    bs, wh, _ = x.shape
    first = (x[:, :, :1] - y) ** 2
    first = first.mean(dim=1, keepdims=True)
    first_sort, first_indices = torch.sort(first, dim=-1, descending=False)
    first_indices = first_indices.repeat([1, wh, 1])

    y_second = torch.gather(y, dim=-1, index=first_indices[:, :, 1:])
    second = (x[:, :, 1:2] - y_second) ** 2
    second = second.mean(dim=1, keepdims=True)
    second_sort, second_indices = torch.sort(second, dim=-1, descending=False)
    second_indices = second_indices.repeat([1, wh, 1])

    y_third = torch.gather(y_second, dim=-1, index=second_indices[:, :, 1:])
    third = (x[:, :, 2:] - y_third) ** 2
    return first_sort[:, :, :1] + second_sort[:, :, :1] + third


def perm_inv_mse_channel_edge(x, y):
    bs, wh, _, n = x.shape
    first = (x[:, :, :1, :] - y) ** 2
    first = first.mean(dim=1, keepdims=True)
    first_sort, first_indices = torch.sort(first, dim=2, descending=False)
    first_indices = first_indices.repeat([1, wh, 1, 1])

    y_second = torch.gather(y, dim=2, index=first_indices[:, :, 1:, :])
    second = (x[:, :, 1:2, :] - y_second) ** 2
    second = second.mean(dim=1, keepdims=True)
    second_sort, second_indices = torch.sort(second, dim=2, descending=False)
    second_indices = second_indices.repeat([1, wh, 1, 1])

    y_third = torch.gather(y_second, dim=2, index=second_indices[:, :, 1:, :])
    third = (x[:, :, 2:, :] - y_third) ** 2
    return first_sort[:, :, :1] + second_sort[:, :, :1] + third


def perm_inv_mse_channel_edge_new(x, y, gt):
    bs, wh, _, n = x.shape
    gt_mean = (gt[:, :, :1, :] - y) ** 2
    y_mean = gt_mean.mean(dim=1, keepdims=True)
    y_sort, y_indices = torch.sort(y_mean, dim=2, descending=False)
    y_indices = y_indices.repeat([1, wh, 1, 1])
    y = torch.gather(y, dim=2, index=y_indices[:, :, 1:, :])

    first = (y[:, :, :1, :] - x) ** 2
    first = first.mean(dim=1, keepdims=True)
    first_sort, first_indices = torch.sort(first, dim=2, descending=False)
    first_indices = first_indices.repeat([1, wh, 1, 1])

    x_second = torch.gather(x, dim=2, index=first_indices[:, :, 1:, :])
    second = (y[:, :, 1:2, :] - x_second) ** 2
    second = second.mean(dim=1, keepdims=True)
    second_sort, second_indices = torch.sort(second, dim=2, descending=False)
    #     second_indices = second_indices.repeat([1, wh, 1, 1])

    #     third = (x[:,:,2:, :] - 1)**2
    return (first_sort[:, :, :1].mean() + second_sort[:, :, :1].mean())  # /y.sum()# + third.mean()


def vis_sdf(xy, dist, axis, img_size=512, plot_points=False):
    xy = xy.cpu().view(1, img_size, img_size, 2).detach().numpy()
    dist = dist.cpu().view(img_size, img_size).detach().numpy()
    # import pdb; pdb.set_trace()
    y, x = xy[0, :, :, 0], xy[0, :, :, 1]

    if plot_points:
        xy_ = model_input.cpu().view(1, 128, 128, 2).detach().numpy()
        y_, x_ = xy_[0, :, :, 0], xy_[0, :, :, 1]
        axis.plot(x_, y_, 'o', color='black')

    axis.contourf(x, y, dist, 16, origin='lower');
    # axis.colorbar();
    axis.contour(x, y, dist, levels=[0.0], colors='white', origin='lower')
    axis.set_ylim(axis.get_ylim()[::-1])


def save_sdf(xy, dist, axis, img_size=512, plot_points=False):
    xy = xy.cpu().view(1, img_size, img_size, 2).detach().numpy()
    dist = dist.cpu().view(img_size, img_size).detach().numpy()
    # import pdb; pdb.set_trace()
    y, x = xy[0, :, :, 0], xy[0, :, :, 1]

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    #     CS = axes.contourf(x, y, dist, 16, origin='lower');
    axes.contour(x, y, dist, levels=[0.0], colors='black')
    axes.set_ylim(axes.get_ylim()[::-1])
    #     axes.axis('off')
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    fig.savefig('image1.svg')


def vis_normal(xy, uv, axis, img_size=512):
    xy = xy.cpu().view(1, img_size, img_size, 2).detach().numpy()
    y, x = xy[0, :, :, 0], xy[0, :, :, 1]
    uv = uv.cpu().view(1, img_size, img_size, 2).detach().numpy()
    v, u = uv[0, :, :, 0], uv[0, :, :, 1]
    axis.quiver(x, y, u, v)
    axis.set_ylim(axis.get_ylim()[::-1])


def gaussian_pyramid_loss(recons, input):
    bs, wh, c = input.shape
    img_size = int(math.sqrt(wh))
    input = input.view(bs, img_size, img_size, c).permute(0, 3, 1, 2)
    recons = recons.view(bs, img_size, img_size, c).permute(0, 3, 1, 2)

    recon_loss = F.mse_loss(recons, input, reduction='none').mean(dim=[1, 2, 3])  # + self.lpips(recons, input)*0.1
    for j in range(2, 5):
        recons = dsample(recons)
        input = dsample(input)
        recon_loss = recon_loss + F.mse_loss(recons, input, reduction='none').mean(dim=[1, 2, 3]) / j
    return recon_loss