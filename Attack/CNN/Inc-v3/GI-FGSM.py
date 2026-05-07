"""Implementation of GI-FGSM attack."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from matplotlib import collections
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
import random
import sys
sys.path.append("../../../")
from function.loader import ImageNet
from function.Normalize import Normalize
from function.attack_methods import DI, gkern
from function.LRS import multi_lrs_inv3
T_kernel = gkern(7, 3)

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='../../../dataset/images.csv',
                    help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='../../../dataset/images',
                    help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='../../../outputs/incv3-GI-FGSM-lrs', help='Source Models.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=15, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum decay factor")
parser.add_argument("--pre_epoch", type=int, default=5, help="Pre-convergence iterations for global search")
parser.add_argument("--s", type=int, default=10, help="Global search factor")
parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
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

def gi_fgsm_attack_oats(model, x, gt):

    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    decay = opt.momentum
    pre_epoch = opt.pre_epoch
    s = opt.s
    momentum = torch.zeros_like(x).detach().cuda()
    delta = torch.zeros_like(x).cuda()
    
    for _ in range(pre_epoch):
        adv_images = x + delta
        adv_images.requires_grad = True
        output = multi_lrs_inv3(model[1], model[0](adv_images))
        loss = F.cross_entropy(output, gt)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = grad + momentum * decay
        delta = delta.detach() + alpha * s * momentum.sign()
        delta = torch.clamp(delta, min=-eps, max=eps)
        delta = torch.clamp(x + delta, min=0, max=1) - x
        delta = delta.detach()

    delta = torch.zeros_like(x).cuda()
    
    for _ in range(num_iter):
        adv_images = x + delta
        adv_images.requires_grad = True
        output = multi_lrs_inv3(model[1], model[0](adv_images))
        loss = F.cross_entropy(output, gt)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = grad + momentum * decay
        delta = delta.detach() + alpha * momentum.sign()
        delta = torch.clamp(delta, min=-eps, max=eps)
        delta = torch.clamp(x + delta, min=0, max=1) - x
        delta = delta.detach()
    
    adv_images = torch.clamp(x + delta, min=0, max=1)
    return adv_images


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + '/' + name)


def main():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    base_model = pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda()
    model = torch.nn.Sequential(Normalize(mean, std), base_model)
    
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        adv_img = gi_fgsm_attack_oats(model, images, gt)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        
        save_image(adv_img_np, images_ID, opt.output_dir)


if __name__ == '__main__':
    main()

