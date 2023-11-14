from PIL import Image
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from config import opt
from tqdm import tqdm


root = Path(opt.dataset_root) / opt.extracted_data_dir / 'train' / 'images'
imglist = list(root.glob('*'))
transform = transforms.Compose([transforms.ToTensor()])

mean_avg = torch.tensor([0., 0., 0.])
std_avg = torch.tensor([0., 0., 0.])
h_max, h_min, h_avg = 0, 0, 0
w_max, w_min, w_avg = 0, 0, 0

for i, f in tqdm(enumerate(imglist)):
    img = Image.open(f)
    h, w = img.height, img.width
    img = transform(img)

    mean, std = img.mean([1, 2]), img.std([1, 2])

    if h > h_max:
        h_max = h
    if w > w_max:
        w_max = w
    if i == 0:
        h_min = h
        w_min = w
    if h < h_min:
        h_min = h
    if w < w_min:
        w_min = w

    mean_avg += mean
    std_avg += std
    h_avg += h
    w_avg += w

mean_avg = mean_avg / (i + 1)
std_avg = std_avg / (i + 1)
h_avg = h_avg / (i + 1)
w_avg = w_avg / (i + 1)

print('---------------')
print(f'avg.mean: {mean_avg} | avg.std: {std_avg}')
print(f'h.min: {h_min} | h.max: {h_max} | h.avg: {h_avg}')
print(f'w.min: {w_min} | w.max: {w_max} | w.avg: {w_avg}')