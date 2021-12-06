
import torch
from PIL import Image
from .utils import get_mgrid
import torchvision
import numpy as np

from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import glob
from skimage.transform import resize


class Dataset_wo_Importance(Dataset):
    def __init__(self, folder, sidelength):
        super().__init__()
        glyph_char = '4'
        self.max_corners = 8
        self.sidelength = sidelength
        self.regex = folder + '/{foldername}/{glyph}/'
        self.imgs_filename = glob.glob(self.regex.format(foldername='render_rgb', glyph=glyph_char)+'*.png')#[:100]
        self.coords = get_mgrid(sidelength, 2).reshape(-1, 2)
        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs_filename)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_filename[idx]).convert('RGB')
        img = self.transform(img).permute(1, 2, 0).view(-1, 3)

        mask_file = self.imgs_filename[idx].replace('render_rgb', 'masks_np').replace('png', 'npy')
        mask_np = np.load(mask_file)
        _, _, n_corners = mask_np.shape
        if n_corners < self.max_corners:
            mask_np_z = np.zeros([256, 256, self.max_corners])
            mask_np_z[:, :, :n_corners] = mask_np
        else:
            mask_np_z = mask_np[:,:,:self.max_corners]
        mask_tensor = torch.from_numpy(mask_np_z.astype(np.float32))
        # masks are reshaped later
        idx = torch.tensor(idx, dtype=torch.long)
        return idx, self.coords, img, mask_tensor


class Dataset_Importance_Family(Dataset):
    def __init__(self, folder, sidelength, max_sampling):
        super().__init__()
        self.glyph_char = 'a'
        self.max_sampling = max_sampling
        self.sidelength = sidelength

        self.regex = folder + '/{foldername}/{glyph}/'
        self.imgs_filename = glob.glob(self.regex.format(foldername='render_rgb', glyph=self.glyph_char)+'*.png')
        self.coords = get_mgrid_np(sidelength).reshape(-1,2)
        self.sidelength = sidelength
        self.glyphs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.num_glyphs = len(self.glyphs)
        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
        ])
        self.corner_sampling = True
        self.one_hot = np.array([0,]*self.num_glyphs)

    def __len__(self):
        return len(self.imgs_filename)*self.num_glyphs

    def make_tensor(self, t, dtype=torch.float32):
        return torch.tensor(t, dtype=dtype)

    def __getitem__(self, idx):
        reminder = idx % self.num_glyphs
        idx = idx // self.num_glyphs
        self.imgs_filename[idx] = self.imgs_filename[idx].replace('/'+self.glyph_char+'/', '/'+self.glyphs[reminder]+'/')

        idx = self.make_tensor(idx, dtype=torch.long)

        img = Image.open(self.imgs_filename[idx]).convert('RGB')
        img_np = np.array(img)/255
        img_np = resize(img_np, (self.sidelength, self.sidelength))
        gt_render = np.mean(img_np, axis=-1)

        e_samples = edge_sample(gt_render.copy())
        num_rand_samples = self.max_sampling - e_samples.shape[0]
        if num_rand_samples>0:
            r_samples = rand_sample(num_rand_samples, self.coords)
            importance_samples_idx = np.concatenate([e_samples, r_samples], axis=0)
        else:
            print('Rand Samples negative ', num_rand_samples)
            importance_samples_idx = np.concatenate([e_samples], axis=0)[:self.max_sampling]
        img_values = img_np[importance_samples_idx[:,0], importance_samples_idx[:,1]]
        img_values_tensor = self.make_tensor(img_values)

        gt_render_values = gt_render[importance_samples_idx[:,0], importance_samples_idx[:,1]]
        gt_render__tensor = self.make_tensor(gt_render_values)

        importance_samples = importance_samples_idx/self.sidelength
        importance_samples = 2*importance_samples-1
        importance_samples_tensor = self.make_tensor(importance_samples)

        glyph_idx = self.one_hot.copy()
        glyph_idx[reminder] = 1
        glyph_idx = self.make_tensor(glyph_idx)
        self.imgs_filename[idx] = self.imgs_filename[idx].replace('/'+self.glyphs[reminder]+'/', '/'+self.glyph_char+'/')
        return idx, importance_samples_tensor, img_values_tensor, torch.tensor(0), gt_render__tensor[:, None], glyph_idx


class Dataset_Importance_Family_Local(Dataset):
    def __init__(self, folder, sidelength, max_sampling):
        super().__init__()
        self.glyph_char = 'a'
        self.max_corners = 12
        self.max_sampling = max_sampling
        self.sidelength = sidelength

        factor = int(sidelength*20/256)
        self.mask_idx = factor*factor*self.max_corners
        self.regex = folder + '/{foldername}/{glyph}/'
        self.imgs_filename = glob.glob(self.regex.format(foldername='render_rgb', glyph=self.glyph_char)+'*.png')
        self.coords = get_mgrid_np(sidelength).reshape(-1,2)
        self.sidelength = sidelength
        self.glyphs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.num_glyphs = len(self.glyphs)
        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
        ])
        self.corner_sampling = True
        self.one_hot = np.array([0,]*self.num_glyphs)

    def __len__(self):
        return len(self.imgs_filename)*self.num_glyphs

    def make_tensor(self, t, dtype=torch.float32):
        return torch.tensor(t, dtype=dtype)

    def __getitem__(self, idx):
        reminder = idx % self.num_glyphs
        idx = idx // self.num_glyphs
        self.imgs_filename[idx] = self.imgs_filename[idx].replace('/'+self.glyph_char+'/', '/'+self.glyphs[reminder]+'/')
        mask_file = self.imgs_filename[idx].replace('render_rgb', 'masks_np').replace('png', 'npy')
        mask_np = np.load(mask_file)
        _, _, n_corners = mask_np.shape
        if n_corners < self.max_corners:
            mask_np_z = np.zeros([256, 256, self.max_corners])
            mask_np_z[:, :, :n_corners] = mask_np
        else:
            mask_np_z = mask_np[:,:,:self.max_corners]
        mask_np_z = resize(mask_np_z, (self.sidelength, self.sidelength))

        if self.corner_sampling:
            c_samples, mask_tensor = importance_sample_corner_mask(mask_np_z, self.mask_idx)
            mask_tensor = torch.from_numpy(mask_tensor.astype(np.float32))
            c_sample_number = c_samples.shape[0]
        else:
            c_samples, mask_tensor = np.empty(shape=[0, 2], dtype=np.int), np.empty(shape=[0, 0])
            c_sample_number = self.mask_idx

        idx = self.make_tensor(idx, dtype=torch.long)

        img = Image.open(self.imgs_filename[idx]).convert('RGB')
        img_np = np.array(img)/255
        img_np = resize(img_np, (self.sidelength, self.sidelength))
        gt_render = np.median(img_np, axis=-1)

        e_samples = edge_sample(gt_render.copy())
        num_rand_samples = self.max_sampling - e_samples.shape[0] - c_sample_number
        if num_rand_samples>0:
            r_samples = rand_sample(num_rand_samples, self.coords)
            importance_samples_idx = np.concatenate([c_samples, e_samples, r_samples], axis=0)
        else:
            print('Rand Samples negative ', num_rand_samples)
            importance_samples_idx = np.concatenate([c_samples, e_samples], axis=0)[:self.max_sampling]
        img_values = img_np[importance_samples_idx[:,0], importance_samples_idx[:,1]]
        img_values_tensor = self.make_tensor(img_values)

        gt_render_values = gt_render[importance_samples_idx[:,0], importance_samples_idx[:,1]]
        gt_render__tensor = self.make_tensor(gt_render_values)

        importance_samples = importance_samples_idx/self.sidelength
        importance_samples = 2*importance_samples-1
        importance_samples_tensor = self.make_tensor(importance_samples)

        glyph_idx = self.one_hot.copy()
        glyph_idx[reminder] = 1
        glyph_idx = self.make_tensor(glyph_idx)
        self.imgs_filename[idx] = self.imgs_filename[idx].replace('/'+self.glyphs[reminder]+'/', '/'+self.glyph_char+'/')
        return idx, importance_samples_tensor, img_values_tensor, mask_tensor[:, None, :], gt_render__tensor[:, None], glyph_idx


def get_mgrid_np(sidelen):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = np.linspace(0, sidelen-1, sidelen, dtype=int)
    y = np.linspace(0, sidelen-1, sidelen, dtype=int)
    xv, yv = np.meshgrid(x, y)
    mgrid = np.stack([xv, yv], axis=-1)
    return mgrid

def edge_sample(img_np):
    img_np[img_np==1] = 0
    max_idx = img_np.shape[0]
    x, y = np.where(img_np>0)
    samples = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            samples.append(np.stack([x+i, y+j], axis=-1))
#     samples.append(np.stack([x, y], axis=-1))
    samples = np.concatenate(samples, axis=0)
    samples_uniq = np.unique(samples, axis=0)
    pos_idx = np.all(samples_uniq>=0, axis=1)
    samples_pos = samples_uniq[pos_idx, :]

    inside_idx = np.all(samples_pos<max_idx, axis=1)
    final_samples = samples_pos[inside_idx, :]
    return final_samples

def rand_sample(n, samples_np):
    max_num = samples_np.shape[0]
    idx = np.random.randint(max_num, size=n)
    return samples_np[idx, :]

def corner_sample(rgb, mask):
    x, y = np.where(img_np>0)
    rgb_sample = rgb[x, y, :]
    samples = np.stack([x, y], axis=-1)
    return rgb_sample, samples

def importance_sampling_create_image(idx, img_values, size = 128):
    idx_np = idx.cpu().numpy()
    img_values_np = img_values.cpu().numpy()
    img_values_np_render = np.median(img_values_np, axis=1)

    idx_int = (((idx_np+1)/2)*size).astype(int)
    img = np.zeros([size, size, 3])
    img[idx_int[:,0], idx_int[:,1], :] = img_values_np

    img_render = np.zeros([size, size, 3])
    img_render[idx_int[:,0], idx_int[:,1], 0] = img_values_np_render
    img_render[idx_int[:,0], idx_int[:,1], 1] = img_values_np_render
    img_render[idx_int[:,0], idx_int[:,1], 2] = img_values_np_render
    return img, img_render

def importance_sample_corner_mask(mask, max_n):
    num_corners = mask.shape[-1]
    x, y, t = np.where(mask>0)
    total_pxs = len(t)
    corner_mask = np.zeros([max_n, num_corners])
    # print(max_n, total_pxs, t)
    # import pdb; pdb.set_trace()
    if total_pxs>max_n:
        indices = np.random.randint(total_pxs, size=max_n)#rand_sample(max_n, np.arange(total_pxs))
        # print(indices.shape)
        corner_mask[np.arange(total_pxs)[:max_n], t[indices]] = 1
        samples = np.stack([x[indices], y[indices]], axis=-1)
    else:
        corner_mask[np.arange(total_pxs), t] = 1
        samples = np.stack([x, y], axis=-1)
    return samples, corner_mask

def interp_mask(mask, size):
    bs, w, h, corners = mask.shape
    mask = mask.permute(0, 3, 1, 2)
    interp_mask = torch.nn.functional.interpolate(mask, size=[size, size], mode='bilinear')
    return interp_mask.permute(0, 2, 3, 1).view(bs,-1, 1, corners)

def get_data(sidelength, device='cuda'):
    img1 = Image.open('./data/temp/0_Akashi.png').convert('RGB')
    img2 = Image.open('./data/temp/0_KABOB.png').convert('RGB')

    transform = Compose([
        Resize(sidelength),
        ToTensor(),
    ])
    img1 = transform(img1).permute(1, 2, 0).view(-1, 3)
    img2 = transform(img2).permute(1, 2, 0).view(-1, 3)
    img = torch.stack([img1, img2], dim=0)
    masks = []
    for i in range(6):
        mask_ = Image.open("./data/temp/0_Akashi_edge_mask-0{}-01.png".format(i + 1)).convert('RGB')
        mask_ = transform(mask_).permute(1, 2, 0)[:, :, :1].view(-1, 1)
        masks.append(mask_)
    masks1 = torch.stack(masks, dim=-1)
    masks = []
    for i in range(6):
        mask_ = Image.open("./data/temp/0_KABOB_edge_mask-0{}-01.png".format(i + 1)).convert('RGB')
        mask_ = transform(mask_).permute(1, 2, 0)[:, :, :1].view(-1, 1)
        masks.append(mask_)
    masks2 = torch.stack(masks, dim=-1)
    masks = torch.stack([masks1, masks2], dim=0)
    return img.to(device), masks.to(device)


def get_pos(sidelength, device='cuda'):
    pos = get_mgrid(sidelength, 2).reshape(-1, 2)
    id1 = torch.tensor([1, 0], dtype=torch.float32)
    id2 = torch.tensor([0, 1], dtype=torch.float32)

    id1 = (id1[None, :]).repeat(sidelength * sidelength, 1)
    id2 = (id2[None, :]).repeat(sidelength * sidelength, 1)

    id1 = torch.cat([pos, id1], dim=-1)
    id2 = torch.cat([pos, id2], dim=-1)

    id_total = torch.stack([id1, id2], dim=0)
    return id_total.to(device)


def interpolate_vectors(v1, v2, n):
    step = (v2 - v1) / (n - 1)
    vecs = []
    for i in range(n):
        vecs.append(v1 + i * step)
    return torch.stack(vecs, dim=0)
