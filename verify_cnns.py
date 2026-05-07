"""Implementation of evaluate attack result."""
import os
import argparse
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from torch.utils.data import DataLoader
import pretrainedmodels
import torchvision.models as models
from function.Normalize import Normalize, TfNormalize
from function.loader import ImageNet

batch_size = 10
input_csv = './dataset/images.csv'
input_dir = 'dataset/images'
adv_dir = 'outputs/incv3-GI-FGSM-lrs'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inception_v4':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v1_50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v1_101':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v1_152':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'vgg16':

        try:
            base_model = models.vgg16(weights="DEFAULT")
        except TypeError:
            base_model = models.vgg16(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    base_model.eval().cuda())
    elif net_name == 'vgg19':

        try:
            base_model = models.vgg19(weights="DEFAULT")
        except TypeError:
            base_model = models.vgg19(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    base_model.eval().cuda())
    elif net_name == 'densenet121':

        try:
            base_model = models.densenet121(weights="DEFAULT")
        except TypeError:
            base_model = models.densenet121(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                    base_model.eval().cuda())
    elif net_name == 'tf_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    else:
        print('Wrong model name!')

    return model


def verify(model_name, path):
    try:

        if model_name in ['resnet_v1_152','vgg16', 'vgg19', 'densenet121']:
            img_size = 224
        else:
            img_size = 299

        model = get_model(model_name, path)

        X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor(), T.Resize(img_size)]))
        data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        sum = 0
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()
            with torch.no_grad():
                sum += (model(images).argmax(1) != (gt)).detach().sum().cpu()

        print(model_name + '  ASR = {:.2%}'.format(sum / 1000.0))

        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"{model_name}  ASR = ERROR: {str(e)}")
        torch.cuda.empty_cache()


def verify_ensmodels(model_name, path):
    try:
        img_size = 299
        model = get_model(model_name, path)

        X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor(), T.Resize(img_size)]))
        data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
        sum = 0
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()
            with torch.no_grad():

                sum += (model(images)[0].argmax(1) != (gt + 1)).detach().sum().cpu()

        print(model_name + '  ASR = {:.2%}'.format(sum / 1000.0))
        

        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"{model_name}  ASR = ERROR: {str(e)}")

        torch.cuda.empty_cache()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_dir', type=str, default='outputs/incv3-MI-lrs')
    parser.add_argument('--input_csv', type=str, default='./dataset/images.csv')
    parser.add_argument('--input_dir', type=str, default='./dataset/images')
    parser.add_argument('--models_path', type=str, default='./models/')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()

    global adv_dir, input_csv, input_dir, batch_size
    adv_dir = args.adv_dir
    input_csv = args.input_csv
    input_dir = args.input_dir
    batch_size = args.batch_size
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print(f"Directory of Adversarial Examples: {adv_dir}")
    print("="*60)
    
    model_names = ['inception_v3', 'inception_v4', 'vgg16', 'vgg19', 'densenet121', 'inc_res_v2', 'resnet_v1_152','resnet_v1_50', 'resnet_v1_101']
    model_names_ens = ['tf_adv_inception_v3', 'tf_ens3_adv_inc_v3',
                       'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']
    models_path = args.models_path
    
    for model_name in model_names:
        verify(model_name, models_path)
        print("="*60)
    for model_name in model_names_ens:
        verify_ensmodels(model_name, models_path)
        print("="*60)


if __name__ == '__main__':
    main()
