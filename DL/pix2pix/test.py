from utils import *
from dataset import *
from model import *

import torch, os, argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='facades')
args = parser.parse_args()


USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

data_dirs = {
  'edge2shoes': '../data/edges2shoes/test',
  'facades': '../data/facades/test',
  'maps': '../data/maps/test',
  'cityscapes': '../data/cityscapes/test'
}

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


data_dir = data_dirs[args.dataset]

data_test = FacadesDataset(data_dir=data_dir, transform=transform)
loader_test = DataLoader(data_test, batch_size=1, shuffle=False)

G = Pix2Pix_G().to(DEVICE)
checkpoint = torch.load(f'{args.dataset}_checkpoint/model_epoch_200.tar')
G.load_state_dict(checkpoint['G_net'])

G.eval()

print('TEST START')
for idx, data in enumerate(loader_test):
  x = data['real'].to(DEVICE)
  y = data['facade'].to(DEVICE)

  test_img = G(y)

  x_ = fn_tonumpy(fn_denorm(x, mean=0.5, std=0.5)).squeeze()
  fake = fn_tonumpy(fn_denorm(test_img, mean=0.5, std=0.5)).squeeze()
  y_ = fn_tonumpy(fn_denorm(y, mean=0.5, std=0.5)).squeeze()

  save_img(x_, fake, y_, f'./{args.dataset}_img', idx+1, test=True)

print('TEST FIN')