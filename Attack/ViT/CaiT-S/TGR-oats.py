
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
from function.LRS import multi_lrs_cait_qkv
from torchvision import transforms as T
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import argparse


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TGR_Attack_LRS:

    def __init__(self, model, epsilon=16/255, steps=10, decay=1.0, 
                 sample_num_batches=130, use_patch_out=False,
                 num_iters=5, shallow_layer=4, balanced_layer=7, deep_layer=10,
                 compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                 compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                 compression_rate_deep=0.0, rank_ratio_deep=0.1):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.sample_num_batches = sample_num_batches
        self.use_patch_out = use_patch_out
        self.image_size = 224
        self.crop_length = 16
        self.max_num_batches = int((224/16)**2)

        self.num_iters = num_iters
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
        
        def attn_tgr(module, grad_in, grad_out, gamma):

            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
                
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if len(grad_in[0].shape) == 4:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H*W)

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = np.clip(max_all // W, 0, H - 1)
                max_all_W = np.clip(max_all % W, 0, W - 1)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = np.clip(min_all // W, 0, H - 1)
                min_all_W = np.clip(min_all % W, 0, W - 1)

                out_grad[:, range(C), max_all_H, :] = 0.0
                out_grad[:, range(C), :, max_all_W] = 0.0
                out_grad[:, range(C), min_all_H, :] = 0.0
                out_grad[:, range(C), :, min_all_W] = 0.0

            
            return (out_grad, )
        
        def v_tgr(module, grad_in, grad_out, gamma):

            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
                
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if len(grad_in[0].shape) == 3:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            if len(grad_in) > 1:
                return (out_grad, grad_in[1])
            else:
                return (out_grad,)
        
        def mlp_tgr(module, grad_in, grad_out, gamma):

            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
                
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if len(grad_in[0].shape) == 3:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] = 0.0
                out_grad[:, min_all, range(c)] = 0.0

            return_dics = (out_grad,)
            for i in range(1, len(grad_in)):
                return_dics = return_dics + (grad_in[i],)
            return return_dics

        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        if hasattr(self.model, 'blocks'):
            for i in range(len(self.model.blocks)):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
    
    def _generate_samples_for_interactions(self, perts, seed):

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
        add_perturbation = perts * add_noise_mask
        return add_perturbation
    
    def _mul_std_add_mean(self, inps, mean, std):
        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        inps.mul_(std_t[:, None, None]).add_(mean_t[:, None, None])
        return inps
    
    def _sub_mean_div_std(self, inps, mean, std):
        """Normalize"""
        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        inps = (inps - mean_t[:, None, None]) / std_t[:, None, None]
        return inps
    
    def _update_perts(self, perts, grad, step_size):
        """Update perturbations"""
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts
    
    def forward(self, inps, labels, mean, std):

        loss_fn = nn.CrossEntropyLoss()
        
        # Clean images in [0, 1] range
        clean_images = inps.clone()
        
        # Initialize perturbations and momentum
        perts = torch.zeros_like(clean_images).cuda()
        perts.requires_grad_()
        momentum = torch.zeros_like(clean_images).cuda()
        
        for i in range(self.steps):
            if self.use_patch_out:
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                normalized_input = self._sub_mean_div_std(
                    clean_images + add_perturbation, mean, std
                )
            else:
                normalized_input = self._sub_mean_div_std(
                    clean_images + perts, mean, std
                )

            outputs = multi_lrs_cait_qkv(
                self.model,
                normalized_input,
                num_iters=self.num_iters,
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

            cost = loss_fn(outputs, labels)

            cost.backward()
            grad = perts.grad.data
            
            # MI-FGSM momentum
            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            perts.data = perts.data + self.step_size * grad.sign()
            perts.data = torch.clamp(perts.data, -self.epsilon, self.epsilon)
            perts.data = torch.clamp(clean_images.data + perts.data, 0.0, 1.0) - clean_images.data
            perts.grad.data.zero_()

        return (clean_images + perts.data).detach()


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
    parser.add_argument('--output_dir', type=str, default='../../../outputs_vit/cait-TGR-lrs')
    parser.add_argument('--max_epsilon', type=float, default=16.0)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--use_patch_out', action='store_true', help='Use PatchOut mechanism')
    parser.add_argument('--sample_num_batches', type=int, default=130, help='Number of patches for PatchOut')
    parser.add_argument('--seed', type=int, default=123)
    

    parser.add_argument('--num_iters', type=int, default=5)
    parser.add_argument('--shallow_layer', type=int, default=4)
    parser.add_argument('--balanced_layer', type=int, default=7)
    parser.add_argument('--deep_layer', type=int, default=10)
    parser.add_argument('--compression_rate_shallow', type=float, default=0.3)
    parser.add_argument('--rank_ratio_shallow', type=float, default=0.01)
    parser.add_argument('--compression_rate_balanced', type=float, default=0.2)
    parser.add_argument('--rank_ratio_balanced', type=float, default=0.04)
    parser.add_argument('--compression_rate_deep', type=float, default=0.0)
    parser.add_argument('--rank_ratio_deep', type=float, default=0.1)
    
    opt = parser.parse_args()
    
    # Set random seed
    set_random_seed(opt.seed)
    
    # Setup
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    vit_model = timm.create_model('cait_s24_224', pretrained=True).eval().cuda()

    attacker = TGR_Attack_LRS(
        model=vit_model,
        epsilon=opt.max_epsilon / 255.0,
        steps=opt.num_iter,
        decay=opt.momentum,
        sample_num_batches=opt.sample_num_batches,
        use_patch_out=opt.use_patch_out,
        num_iters=opt.num_iters,
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

        adv_images = attacker.forward(images, gt, mean, std)

        adv_img_np = adv_images.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        
        save_image(adv_img_np, images_ID, opt.output_dir)
    
    print(f"Done! Adversarial images saved to {opt.output_dir}")


if __name__ == '__main__':
    main()

