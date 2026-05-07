import sys
import os
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import timm
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from function.loader import ImageNet


BASIC_MODELS = [
    'pit_s_224',
    'cait_s24_224',
    'deit_base_patch16_224',
    'swin_small_patch4_window7_224',
    'vit_base_patch16_224',
    'visformer_small',
    'convit_base',
    'twins_pcpvt_base',
]

ADVANCED_MODELS = [
    'deit_small_patch16_224',
    'swin-Badv',
    'xcit_small_12_p16_224',
 ]


models = BASIC_MODELS

def create_timm_model(model_name):

    from torch import nn
    from torchvision import transforms as T
    from function.Normalize import Normalize

    if model_name == 'xcit_small_12_p16_224':
        base_model = timm.create_model(model_name, pretrained=False)
        model_path = './model_tf/xcit-s12-ImageNet-eps-8.pth.tar'
        if os.path.exists(model_path):
            base_model.load_state_dict(torch.load(model_path))
        else:
            print(f"Warning: Model file not found: {model_path}, using randomly initialized weights")

        model = nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            base_model
        ).eval().cuda()
        
    elif model_name == 'deit_small_patch16_224':
        base_model = timm.create_model(model_name, pretrained=False)
        model_path = './model_tf/advdeit_small.pth'
        if os.path.exists(model_path):
            base_model.load_state_dict(torch.load(model_path)['model'])
        else:
            print(f"Warning: Model file not found: {model_path}, using randomly initialized weights")

        model = nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            base_model
        ).eval().cuda()
        
    elif model_name == 'swin-Badv':
        try:
            from rebuttal_model.swin import swin_base_patch4_window7_224

            model_path = './model_tf/ad-Swin-B.pth'

            base_model = swin_base_patch4_window7_224(pretrained=False)
            
            if os.path.exists(model_path):
                a = torch.load(model_path, map_location='cuda:0')
                new_state_dict = {k.replace('module.', ''): v for k, v in a['state_dict'].items()}
                base_model.load_state_dict(new_state_dict)
            else:
                print(f"Warning: Model file not found: {model_path}, using randomly initialized weights")

            model = nn.Sequential(
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                base_model
            ).eval().cuda()
            
        except ImportError as e:
            print(f"Error: Failed to import swin model: {e}")
            print("Please make sure rebuttal_model.swin module exists")
            return None, None
        except Exception as e:
            print(f"Error: Failed to load swin-Badv model: {e}")
            return None, None
        
    else:

        try:
            base_model = timm.create_model(model_name, pretrained=True)

            model = nn.Sequential(
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                base_model
            ).eval().cuda()
            
        except Exception as e:
            print(f"Error: Failed to create model {model_name}: {e}")
            return None, None

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    return model, transform


def load_labels(csv_file):

    import csv
    label_dict = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'TrueLabel' in row and 'ImageId' in row:
                image_id = row['ImageId']
                true_label = int(row['TrueLabel']) - 1
                label_dict[image_id] = true_label
            else:
                raise ValueError("CSV file must contain 'ImageId' and 'TrueLabel' columns")

    return label_dict


def verify_images(model, image_path, csv_path, transform, batch_size=20):

    dataset = ImageNet(image_path, csv_path, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=8)
    
    success_count = 0
    total_count = len(dataset)
    
    with torch.no_grad():
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            success_count += (outputs.argmax(1) != gt).detach().sum().cpu().item()
    
    attack_success_rate = success_count / total_count if total_count > 0 else 0.0
    return attack_success_rate, success_count, total_count


def verify_single_model(model_name, adv_path, csv_path, batch_size=20):
    """Verify attack success rate for a single model"""
    model, transform = create_timm_model(model_name)
    if model is None:
        return None
    asr, success, total = verify_images(model, adv_path, csv_path, transform, batch_size)
    print(f"{model_name} ASR: {asr:.2%}")
    del model
    torch.cuda.empty_cache()
    return asr


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_path', type=str, default='./outputs_vit/cait-PNA-lrs')
    parser.add_argument('--labels_path', type=str, default='./dataset/images.csv')
    parser.add_argument('--models', type=str, default='all',
                        choices=['basic', 'advanced', 'all'])
    parser.add_argument('--output_vits_csv', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=20)
    
    opt = parser.parse_args()

    if not os.path.exists(opt.adv_path):
        print(f"Error: Adversarial examples directory does not exist: {opt.adv_path}")
        return
    
    if not os.path.exists(opt.labels_path):
        print(f"Error: Label file does not exist: {opt.labels_path}")
        return

    global models
    if opt.models == 'basic':
        models = BASIC_MODELS
    elif opt.models == 'advanced':
        models = ADVANCED_MODELS
    elif opt.models == 'all':
        models = BASIC_MODELS + ADVANCED_MODELS

    results = {}
    for model_name in models:
        asr = verify_single_model(model_name, opt.adv_path, opt.labels_path, opt.batch_size)
        if asr is not None:
            results[model_name] = asr

    if results:

        avg_asr = np.mean(list(results.values()))
        print(f"AVERAGE ASR: {avg_asr:.2%}")

        if opt.output_vits_csv:
            import csv
            with open(opt.output_vits_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Model Name', 'Attack Success Rate'])
                for model_name, asr in results.items():
                    writer.writerow([model_name, f"{asr:.4f}"])
                writer.writerow(['Average', f"{avg_asr:.4f}"])
    else:
        print("No successful evaluation results")


if __name__ == '__main__':
    main()

