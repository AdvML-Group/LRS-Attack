import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random

from torchvision import transforms
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


class ATT_Attack_LRS:

    
    def __init__(self, model, model_name='cait_s24_224', epsilon=16/255, 
                 steps=10, decay=1.0, lam=0.01, use_patch_out=False,
                 num_iters=5, shallow_layer=4, balanced_layer=7, deep_layer=10,
                 compression_rate_shallow=0.3, rank_ratio_shallow=0.01,
                 compression_rate_balanced=0.2, rank_ratio_balanced=0.04,
                 compression_rate_deep=0, rank_ratio_deep=0.1):
        self.model = model
        self.model_name = model_name
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay
        self.lam = lam
        self.use_patch_out = use_patch_out

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
        
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = 130
        self.max_num_batches = int((224/16)**2)

        self.im_fea = None
        self.im_grad = None
        self.size = 16
        self.patch_index = self.Patch_index(self.size)

        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        self.gamma = 0.5

        self._set_model_params()

        self._register_model()
    
    def _set_model_params(self):

        if self.model_name == 'cait_s24_224':
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
        else:
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
    
    def TR_01_PC(self, num, length):

        rate_l = num
        tensor = torch.cat((torch.ones(rate_l), torch.zeros(length - rate_l)))
        return tensor
    
    def Patch_index(self, size):

        img_size = 224
        filterSize = size
        stride = size
        P = np.floor((img_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P
        index = np.ones([P * Q, filterSize * filterSize], dtype=int)
        tmpidx = 0
        for q in range(Q):
            plus1 = q * stride * img_size
            for p in range(P):
                plus2 = p * stride
                index_ = np.array([], dtype=int)
                for i in range(filterSize):
                    plus = i * img_size + plus1 + plus2
                    index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
                index[tmpidx] = index_
                tmpidx += 1
        index = torch.LongTensor(np.tile(index, (1, 1, 1))).cuda()
        return index
    
    def norm_patchs(self, GF, index, patch, scale, offset):

        patch_size = patch ** 2
        for i in range(len(GF)):
            tmp = torch.take(GF[i], index[i])
            norm_tmp = torch.mean(tmp, dim=-1)
            scale_norm = scale * ((norm_tmp - norm_tmp.min()) / (norm_tmp.max() - norm_tmp.min())) + offset
            tmp_bi = torch.as_tensor(scale_norm.repeat_interleave(patch_size)) * 1.0
            GF[i] = GF[i].put_(index[i], tmp_bi)
        return GF
    
    def _register_model(self):

        
        def attn_ATT(module, grad_in, grad_out):
            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
            

            if self.back_attn < 0 or self.back_attn >= len(self.truncate_layers):
                return grad_in
            
            mask = torch.ones_like(grad_in[0]) * self.truncate_layers[self.back_attn] * self.weaken_factor[0]
            out_grad = mask * grad_in[0][:]
            
            if self.var_A != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            
            if len(grad_in[0].shape) == 4:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H*W)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = np.clip(max_all // W, 0, H-1)
                max_all_W = np.clip(max_all % W, 0, W-1)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = np.clip(min_all // W, 0, H-1)
                min_all_W = np.clip(min_all % W, 0, W-1)
                
                out_grad[:, range(C), max_all_H, :] *= GPF_
                out_grad[:, range(C), :, max_all_W] *= GPF_
                out_grad[:, range(C), min_all_H, :] *= GPF_
                out_grad[:, range(C), :, min_all_W] *= GPF_
            
            self.var_A = torch.var(out_grad)
            self.back_attn -= 1
            return (out_grad, )
        
        def v_ATT(module, grad_in, grad_out):
            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
            
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            
            if self.var_qkv != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            
            if len(grad_in[0].shape) == 3:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                
                out_grad[:, max_all, range(c)] *= GPF_
                out_grad[:, min_all, range(c)] *= GPF_
            
            self.var_qkv = torch.var(out_grad)
            
            if len(grad_in) > 1:
                return (out_grad, grad_in[1])
            else:
                return (out_grad,)
        
        def mlp_ATT(module, grad_in, grad_out):
            if len(grad_in) == 0 or grad_in[0] is None:
                return grad_in
            
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad = mask * grad_in[0][:]
            
            if self.var_mlp != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_mlp))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            
            if len(grad_in[0].shape) == 3:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                
                out_grad[:, max_all, range(c)] *= GPF_
                out_grad[:, min_all, range(c)] *= GPF_
            
            self.var_mlp = torch.var(out_grad)
            
            return_dics = (out_grad,)
            for i in range(1, len(grad_in)):
                return_dics = return_dics + (grad_in[i],)
            return return_dics
        
        def get_fea(module, input, output):
            self.im_fea = output.clone()
        
        def get_grad(module, input, output):
            self.im_grad = output[0].clone()
        
        if hasattr(self.model, 'blocks'):
            num_blocks = len(self.model.blocks)
            if num_blocks >= 11:
                self.get_fea_hook = self.model.blocks[10].register_forward_hook(get_fea)
                self.get_grad_hook = self.model.blocks[10].register_backward_hook(get_grad)
            
            for i in range(num_blocks):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_ATT)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_ATT)
                self.model.blocks[i].mlp.register_backward_hook(mlp_ATT)
    
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
    
    def _sub_mean_div_std(self, inps, mean, std):

        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        inps = (inps - mean_t[:, None, None]) / std_t[:, None, None]
        return inps
    
    def _mul_std_add_mean(self, inps, mean, std):

        dtype = inps.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype).cuda()
        std_t = torch.as_tensor(std, dtype=dtype).cuda()
        inps.mul_(std_t[:, None, None]).add_(mean_t[:, None, None])
        return inps
    
    def _update_perts(self, perts, grad, step_size):

        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts
    
    def forward(self, inps, labels, mean, std):

        loss_fn = nn.CrossEntropyLoss()
        

        clean_images = inps.clone()

        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        if self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        
        normalized_input = self._sub_mean_div_std(clean_images, mean, std)
        output = self.model(normalized_input)
        output.backward(torch.ones_like(output))
        

        resize = transforms.Resize((224, 224))
        if self.model_name == 'cait_s24_224':

            GF = (self.im_fea[0] * self.im_grad[0]).sum(-1)
            GF = resize(GF.reshape(1, 14, 14))
        else:
            try:
                GF = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
                GF = resize(GF.reshape(1, 14, 14))
            except:
                GF = torch.ones(1, 224, 224).cuda()

        GF_patchs_t = self.norm_patchs(GF, self.patch_index, self.size, self.scale, self.offset)
        GF_patchs_start = torch.ones_like(GF_patchs_t).cuda() * 0.99
        GF_offset = (GF_patchs_start - GF_patchs_t) / self.steps
        

        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        if self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
        

        perts = torch.zeros_like(clean_images).cuda()
        perts.requires_grad_()
        momentum = torch.zeros_like(clean_images).cuda()
        
        for i in range(self.steps):

            self.var_A = 0
            self.var_qkv = 0
            self.var_mlp = 0
            if self.model_name == 'cait_s24_224':
                self.back_attn = 24
            elif self.model_name == 'cait_s24_224':
                self.back_attn = 24
            elif self.model_name == 'cait_s24_224':
                self.back_attn = 24
            elif self.model_name == 'cait_s24_224':
                self.back_attn = 24

            torch.manual_seed(i)
            random_patch = torch.rand(14, 14).repeat_interleave(16).reshape(14, 14*16).repeat(1, 16).reshape(224, 224).cuda()
            GF_patchs = torch.where(torch.as_tensor(random_patch > GF_patchs_start - GF_offset * (i + 1)), 0., 1.).cuda()
            

            if self.use_patch_out:
                add_perturbation = self._generate_samples_for_interactions(perts, i)
                normalized_input = self._sub_mean_div_std(
                    clean_images + add_perturbation * GF_patchs.detach(), mean, std
                )
            else:
                normalized_input = self._sub_mean_div_std(
                    clean_images + perts * GF_patchs.detach(), mean, std
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

            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(clean_images.data + perts.data, 0.0, 1.0) - clean_images.data
            perts.grad.data.zero_()

        return (clean_images + perts.data).detach()


def save_image(images, names, output_dir):
    """Save adversarial images"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(os.path.join(output_dir, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='../../../dataset/images.csv')
    parser.add_argument('--input_dir', type=str, default='../../../dataset/images')
    parser.add_argument('--output_dir', type=str, default='../../../outputs_vit/cait-ATT-lrs')
    parser.add_argument('--model_name', type=str, default='cait_s24_224')
    parser.add_argument('--max_epsilon', type=float, default=16.0)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--use_patch_out', action='store_true')
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
    parser.add_argument('--seed', type=int, default=123)
    
    opt = parser.parse_args()

    set_random_seed(opt.seed)
    

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print(f"Loading {opt.model_name} model...")
    vit_model = timm.create_model(opt.model_name, pretrained=True).eval().cuda()

    attacker = ATT_Attack_LRS(
        model=vit_model,
        model_name=opt.model_name,
        epsilon=opt.max_epsilon / 255.0,
        steps=opt.num_iter,
        decay=opt.momentum,
        lam=opt.lam,
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

