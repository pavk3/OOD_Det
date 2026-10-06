#Adapted from: https://github.com/y0ast/Glow-PyTorch/blob/master/Sample_from_Glow.ipynb
import json

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow

device = torch.device("cuda")

output_folder = 'output/'
model_name = 'glow_model_250.pth'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)
    
image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model, n=1024, batch=32):
    imgs, raws = [], []
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            if hparams['y_condition']:
                y = torch.eye(num_classes).repeat(batch // num_classes + 1, 1)
                y = y[:batch, :].to(device)
            else:
                y = None
            x_raw = model(y_onehot=y, temperature=1, reverse=True)
            raws.append(x_raw[:k].cpu())
            #imgs.append(postprocess(x_raw[:k]).cpu())
    return torch.cat(imgs), torch.cat(raws)


def compute_nll(dataset, model):
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=4)
    nlls = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device) if hparams['y_condition'] else None
        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)
        nlls.append(nll.cpu())
    return torch.cat(nlls)

images, x_raw = sample(model, n=1024)
gen_ds = TensorDataset(x_raw, torch.zeros(len(x_raw), dtype=torch.long))

gen_nll   = compute_nll(gen_ds, model)
cifar_nll = compute_nll(test_cifar, model)

plt.figure(figsize=(12, 6))
plt.hist(-cifar_nll.numpy(), bins=50, density=True, alpha=0.6, label="CIFAR-10 test")
plt.hist(-gen_nll.numpy(),   bins=50, density=True, alpha=0.6, label="Glow samples")
plt.xlabel("Negative bits per dimension")
plt.legend()
plt.show()


