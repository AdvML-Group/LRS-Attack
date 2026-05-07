
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
from functools import partial
import sys
sys.path.append("../../../")
from function.loader import ImageNet
from function.Normalize import Normalize
from torchvision import transforms as T
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import argparse
from function.LRS import multi_lrs_cait_qkv

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PNA_Attack_LRS:
    
    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, 
                 sample_num_batches=130, lamb=0.1,
                 shallow_layer=4, balanced_layer=7, deep_layer=10,
                 compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                 compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                 compression_rate_deep=0.0, rank_ratio_deep=0.1):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.lamb = lamb
        self.sample_num_batches = sample_num_batches
        self.image_size = 224
        self.crop_length = 16
        self.max_num_batches = int((224/16)**2)
        

        self.shallow_layer = shallow_layer
        self.balanced_layer = balanced_layer
        self.deep_layer = deep_layer
        self.compression_rate_shallow = compression_rate_shallow
        self.rank_ratio_shallow = rank_ratio_shallow
        self.compression_rate_balanced = compression_rate_balanced
        self.rank_ratio_balanced = rank_ratio_balanced
        self.compression_rate_deep = compression_rate_deep
        self.rank_ratio_deep = rank_ratio_deep
        
        self._register_model()
    
    def _register_model(self):
        """Register backward hooks"""
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )
        
        drop_hook_func = partial(attn_drop_mask_grad, gamma=0)
        
        if hasattr(self.model, 'blocks'):
            for i in range(len(self.model.blocks)):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)
    
    def _generate_samples_for_interactions(self, perts, seed):
        """Generate patch-based perturbation masks"""
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size / self.crop_length)
        
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])
        
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        for r, c in zip(rows, cols):
            add_noise_mask[:, :, r*self.crop_length:(r+1)*self.crop_length, 
                          c*self.crop_length:(c+1)*self.crop_length] = 1
        return perts * add_noise_mask
    
    def _mul_std_add_mean(self, inps, mean, std):
        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        inps.mul_(std_t[:, None, None]).add_(mean_t[:, None, None])
        return inps
    
    def _sub_mean_div_std(self, inps, mean, std):
        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        return (inps - mean_t[:, None, None]) / std_t[:, None, None]
    
    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        return torch.clamp(perts, -self.epsilon, self.epsilon)
    
    def forward(self, inps, labels, mean, std):

        loss_fn = nn.CrossEntropyLoss()
        unnorm_inps = self._mul_std_add_mean(inps.clone(), mean, std)
        
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()
        momentum = torch.zeros_like(inps).cuda()
        
        for i in range(self.steps):
            add_perturbation = self._generate_samples_for_interactions(perts, i)
            perturbed_input = self._sub_mean_div_std(
                unnorm_inps + add_perturbation, mean, std
            )
            

            outputs = multi_lrs_cait_qkv(
                self.model, 
                perturbed_input,
                num_iters=5,
                shallow_layer=self.shallow_layer,
                balanced_layer=self.balanced_layer,
                deep_layer=self.deep_layer,
                compression_rate_shallow=self.compression_rate_shallow,
                rank_ratio_shallow=self.rank_ratio_shallow,
                compression_rate_balanced=self.compression_rate_balanced,
                rank_ratio_balanced=self.rank_ratio_balanced,
                compression_rate_deep=self.compression_rate_deep,
                rank_ratio_deep=self.rank_ratio_deep
            )
            
            cost1 = loss_fn(outputs, labels)
            cost2 = torch.norm(perts)
            cost = cost1 + self.lamb * cost2
            
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        
        adv_inps = self._sub_mean_div_std(unnorm_inps + perts.data, mean, std)
        return adv_inps.detach()


def save_image(images, names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(os.path.join(output_dir, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='../../../dataset/images.csv')
    parser.add_argument('--input_dir', type=str, default='../../../dataset/images')
    parser.add_argument('--output_dir', type=str, default='../../../outputs_vit/cait-PNA-lrs')
    parser.add_argument('--max_epsilon', type=float, default=16.0)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--lamb', type=float, default=0.1)
    parser.add_argument('--sample_num_batches', type=int, default=130)
    parser.add_argument('--shallow_layer', type=int, default=4)
    parser.add_argument('--balanced_layer', type=int, default=7)
    parser.add_argument('--deep_layer', type=int, default=10)
    parser.add_argument('--compression_rate_shallow', type=float, default=0.3)
    parser.add_argument('--rank_ratio_shallow', type=float, default=0.01)
    parser.add_argument('--compression_rate_balanced', type=float, default=0.2)
    parser.add_argument('--rank_ratio_balanced', type=float, default=0.04)
    parser.add_argument('--compression_rate_deep', type=float, default=0.0)
    parser.add_argument('--rank_ratio_deep', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    
    opt = parser.parse_args()
    set_random_seed(opt.seed)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    deit_model = timm.create_model('cait_s24_224', pretrained=True).eval().cuda()
    
    attacker = PNA_Attack_LRS(
        model=deit_model,
        epsilon=opt.max_epsilon / 255.0,
        steps=opt.num_iter,
        decay=opt.momentum,
        sample_num_batches=opt.sample_num_batches,
        lamb=opt.lamb,
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
    
    transforms = T.Compose([T.CenterCrop(224), T.ToTensor()])
    dataset = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, 
                            pin_memory=True, num_workers=8)

    for images, images_ID, gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        
        # Normalize
        normalize = Normalize(mean, std)
        normalized_images = normalize(images)
        
        # Attack
        adv_images = attacker.forward(normalized_images, gt, mean, std)

        adv_images_denorm = adv_images.clone()
        for i in range(3):
            adv_images_denorm[:, i] = adv_images_denorm[:, i] * std[i] + mean[i]
        
        adv_img_np = adv_images_denorm.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)
    
    print(f"Done! Saved to {opt.output_dir}")


if __name__ == '__main__':
    main()

