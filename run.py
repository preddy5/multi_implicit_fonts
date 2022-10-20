
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

from implicit_fonts.argparser import args
from implicit_fonts.io import Dataset_Importance_Family_Local, Dataset_Importance_Family,\
    importance_sampling_create_image
from implicit_fonts.models import AutoDecoderFamily, gradient
from implicit_fonts.utils import render_sdf, select_top2, perm_inv_mse_channel_edge_new, fig2data
from shutil import copytree, ignore_patterns, rmtree
import click

exec('from implicit_fonts.configs.{} import *'.format(args.config))
torch.manual_seed(42)
device = 'cuda'
local_supervision = args.local_supervision
change_channels = []#[0, 1, 2, 3]
ratio = 1.0


if local_supervision:
    dataset = Dataset_Importance_Family_Local(folder=dataset_folder, sidelength=img_size, max_sampling=max_sampling)
    mask_idx = dataset.mask_idx
else:
    dataset = Dataset_Importance_Family(folder=dataset_folder, sidelength=img_size, max_sampling=max_sampling)


dataloader = DataLoader(dataset, batch_size=batchsize,
                        shuffle=True, num_workers=32)

implicit_global = AutoDecoderFamily(in_features=2, out_features=3, hidden_features=384,
                                     hidden_layers=6, outermost_linear=True, first_omega_0=1, hidden_omega_0=1,
                                    num_embed=len(dataset)//dataset.num_glyphs, num_glyphs=dataset.num_glyphs)
implicit_global.to(device)

params = list(implicit_global.parameters())
optim = torch.optim.Adam(lr=1e-3, params=params)
init_epoch = 0

create_new= True
if os.path.exists(save_dir.format(args.version, '')):
    if click.confirm('Folder exists do you want to override?', default=True):
        rmtree(save_dir.format(args.version, ''))
    else:
        checkpoint = torch.load(save_dir.format(args.version, 'checkpoints/')+'checkpoint_generic.pt')
        implicit_global.load_state_dict(checkpoint['model_params'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['init_epoch']
        create_new = False
if create_new:
    os.makedirs(save_dir.format(args.version, ''))
    os.makedirs(save_dir.format(args.version, 'checkpoints'))
    os.makedirs(save_dir.format(args.version, 'images'))
    copytree('./', os.path.join(save_dir.format(args.version, ''), 'code'),
         ignore=ignore_patterns('*.pyc', 'tmp*', 'logs*', 'data*', 'experiment_scripts*', 'notebook*', '.idea*'))


loss_cache = []
total_steps = 0
corner_recon_loss = torch.tensor(0)
phrase = 'generic'
if init_epoch < init_training:
    dataloader.dataset.corner_sampling = False

for i_epoch in range(init_epoch, epochs):
    loss_print = 0
    for i_batch, sample_batched in enumerate(dataloader):
        if i_epoch>1000:
            implicit_global.embed.weight.requires_grad = False

        idx, coords, gt_image, corner_mask, gt_render, glyph_idx = [i.to(device) for i in sample_batched]

        [model_output_global, output_coords], label_embed = implicit_global([idx, glyph_idx, coords])
        model_output_global = F.tanh(model_output_global)

        if init_training>0:
            alias_warmup = 1 - ((1-4/(img_size))*i_epoch)/10
        else:
            alias_warmup = 0
        model_cat = render_sdf(model_output_global,
                               max(alias_warmup, 4/(img_size)))

        if i_epoch <= init_training:
            img_grad = gradient(torch.mean(model_output_global, dim=-1, keepdims=True), output_coords)
            final_render = torch.mean(model_cat, dim=-1, keepdims=True)
        else:
            rgb, indices = select_top2(model_cat, True)
            sdf_gather = torch.gather(model_output_global, dim=-1, index=indices)
            img_grad = gradient(torch.mean(sdf_gather, dim=-1, keepdims=True), output_coords)
            final_render = torch.mean(rgb, dim=-1, keepdims=True)

        image_loss = (final_render- gt_render)**2
        grad_loss = torch.abs(torch.clamp(img_grad.norm(dim=-1), min=-1, max=1) - 1).mean()
        # ------------------------------------------------------------------------

        if i_epoch <= init_training or not local_supervision:
            loss = image_loss.mean() + grad_loss * 0.01
        else:
            corner = model_cat[:,:mask_idx,:,None]*corner_mask
            gt_corner = gt_image[:,:mask_idx,:,None]*corner_mask
            corner_recon_loss = perm_inv_mse_channel_edge_new(corner, gt_corner, gt_render[:,:mask_idx,:,None]*corner_mask).sum()
            loss = image_loss.mean() + grad_loss * 0.01 + corner_recon_loss*4
        loss += 1e-4 * torch.mean(label_embed.pow(2))
        loss_print +=loss
        if not i_batch % steps_til_summary:
            loss_cache.append([grad_loss.cpu().detach().numpy().item(),
                               corner_recon_loss.cpu().detach().numpy().item(),
                               loss.cpu().detach().numpy().item()])
            print("Epoch %d, Total loss %0.6f, img loss %0.6f, grad loss %0.6f , corner loss %0.6f"
                  % (i_epoch, loss_print/(1+i_batch), image_loss.mean(), grad_loss * 0.01, corner_recon_loss))

            fig, axes = plt.subplots(1, 3, figsize=(4, 2))
            img_rgb_gt, img_render_gt = importance_sampling_create_image(coords[0], gt_image[0], img_size)
            img_rgb, img_render = importance_sampling_create_image(coords[0], model_cat[0].clone().detach(), img_size)
            axes[0].imshow(img_rgb_gt)
            axes[0].axis('off')
            axes[1].imshow(img_rgb)
            axes[1].axis('off')
            axes[2].imshow(img_render)
            axes[2].axis('off')
            plt.imsave(save_dir.format(args.version, 'images/{}_{}.png').format(i_epoch, i_batch), fig2data(fig))
            plt.close()

        if i_epoch ==init_training:
            phrase = str(init_training)
            dataloader.dataset.corner_sampling = True
        elif i_epoch % 10 ==0:
            phrase = str(i_epoch)
        else:
            phrase = 'generic'

        optim.zero_grad()
        loss.backward()
        optim.step()
        total_steps +=1
    torch.save({
        'model_params': implicit_global.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'init_epoch': i_epoch,
            }, save_dir.format(args.version, 'checkpoints/')+'checkpoint_{}.pt'.format(phrase))
    np.savetxt(save_dir.format(args.version, 'checkpoints/')+ 'loss.csv', np.array(loss_cache), delimiter=',')
print(loss_cache)
