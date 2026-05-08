
import os
import timm

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from matplotlib import collections
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import sys
import random

sys.path.append("../../../")
from function.loader import ImageNet
from function.Normalize import Normalize
from function.attack_methods import DI, gkern
from function.LRS import multi_lrs_cait_qkv

T_kernel = gkern(7, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='../../../dataset/images.csv',
                    help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='../../../dataset/images',
                    help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='../../../outputs_vit/cait-MI-lrs-qkv', help='Output directory.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=224, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=224, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=5, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--compression_rate", type=float, default=0.7)
parser.add_argument("--rank_ratio", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
parser.add_argument("--shallow_layer", type=int, default=4 )
parser.add_argument("--balanced_layer", type=int, default=7 )
parser.add_argument("--deep_layer", type=int, default=10)
parser.add_argument("--compression_rate_shallow", type=float, default=0.3)
parser.add_argument("--rank_ratio_shallow", type=float, default=0.01)
parser.add_argument("--compression_rate_balanced", type=float, default=0.2)
parser.add_argument("--rank_ratio_balanced", type=float, default=0.04)
parser.add_argument("--compression_rate_deep", type=float, default=0.0)
parser.add_argument("--rank_ratio_deep", type=float, default=0.1)

opt = parser.parse_args()
torch.backends.cudnn.benchmark = True

transforms = T.Compose([T.CenterCrop(opt.image_width), T.ToTensor()])


torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
os.environ['PYTHONHASHSEED'] = str(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def clip_by_tensor(t, t_min, t_max):

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def graph(model, x, gt):
    eps = opt.max_epsilon / 255.0
    num_iter = 10
    alpha = eps / num_iter

    decay = 1.0
    adv_images = x.clone().detach()
    momentum = torch.zeros_like(x).detach().cuda()

    for i in range(num_iter):
        adv_images.requires_grad = True
        output = multi_lrs_cait_qkv(
            model[1],
            model[0](adv_images),
            num_iters=5,
            shallow_layer=opt.shallow_layer,
            balanced_layer=opt.balanced_layer,
            deep_layer=opt.deep_layer,
            compression_rate_shallow=opt.compression_rate_shallow,
            rank_ratio_shallow=opt.rank_ratio_shallow,
            compression_rate_balanced=opt.compression_rate_balanced,
            rank_ratio_balanced=opt.rank_ratio_balanced,
            compression_rate_deep=opt.compression_rate_deep,
            rank_ratio_deep=opt.rank_ratio_deep
        )

        loss = F.cross_entropy(output, gt)

        grad = torch.autograd.grad(loss, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        # MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - x, min=-eps, max=eps)
        adv_images = torch.clamp(x + delta, min=0, max=1).detach()

    return adv_images


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + '/' + name)


def main():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    deit_model = timm.create_model('cait_s24_224', pretrained=True).eval().cuda()

    model = torch.nn.Sequential(
        Normalize(mean, std),
        deit_model
    )
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        adv_img = graph(model, images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255

        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()

