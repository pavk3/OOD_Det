#Adapted from: https://github.com/y0ast/Glow-PyTorch/blob/master/Sample_from_Glow.ipynb
import json
import math
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import numpy as np
from scipy.stats import gaussian_kde

from datasets import get_CIFAR10, get_SVHN, postprocess
from model import Glow

#Slight improvement with more samples (1K vs 10K).

device = torch.device("cuda")

output_folder = 'glow/'
model_name = 'glow_affine_coupling.pt'

with open(output_folder + 'hparams.json') as json_file:  
    hparams = json.load(json_file)

#Below line modify hparams with updated data path.
hparams['dataroot'] = './data'  
image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], hparams['download'])
image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], hparams['download'])


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])

model.load_state_dict(torch.load(output_folder + model_name))
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

no=10000
def sample(model, n=no, batch=32):
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
            imgs.append(postprocess(x_raw[:k]).cpu())
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

images, x_raw = sample(model, n=no)

#Quantize the samples to apply same transformation as preprocess does on test data.
#Not much of an effect
def quantize(x):
    x = torch.clamp(x, -0.5, 0.5)
    x = torch.floor((x + 0.5) * 256).clamp(0, 255)
    return x / 256 - 0.5

x_raw_q = quantize(x_raw)
mask = torch.isfinite(x_raw_q).all(dim=(1, 2, 3))
x_raw_q = x_raw_q[mask]
#print(f"dropped {(~mask).sum().item()} of {len(mask)} samples")
gen_ds = TensorDataset(x_raw_q, torch.zeros(len(x_raw_q), dtype=torch.long))
#print(torch.isnan(x_raw).float().mean().item())      # all, or some?
#print(torch.isnan(x_raw).any(dim=(1,2,3)).sum().item())  # how many images
#print(sum(torch.isnan(p).sum().item() for p in model.parameters()))

gen_nll   = compute_nll(gen_ds, model)
cifar_nll = compute_nll(test_cifar, model)
svhn_nll = compute_nll(test_svhn, model)
print(f"Gen nll: {gen_nll.mean():.3f}, CIFAR-10 nll: {cifar_nll.mean():.3f}, SVHN_nll: {svhn_nll.mean():.3f}")
cifar = -cifar_nll.numpy() * np.log(2.0) * 3072
svhn = -svhn_nll.numpy() * np.log(2.0) * 3072
gen   = -gen_nll.numpy()   * np.log(2.0) * 3072

xs = np.linspace(min(cifar.min(), svhn.min(), gen.min()), max(cifar.max(), svhn.max(), gen.max()), 500)

plt.figure(figsize=(8, 5))
for data, label, colour in [(cifar, "CIFAR-10 test (Real)", "C0"),
                            (svhn,   "SVHN test (Real)",  "C1"),
                            (gen,   "Glow samples (Generated)",  "C2")]:
    plt.hist(data, bins=50, density=True, alpha=0.35, color=colour)
    plt.plot(xs, gaussian_kde(data)(xs), color=colour, lw=2, label=label)

plt.xlabel("Log-likelihood (nats)")
plt.ylabel("Density")
plt.legend()
plt.title(f"Log-likelihood plot for {no} samples")
plt.tight_layout()
plt.savefig("hist.png", dpi=150)


