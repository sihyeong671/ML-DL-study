import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

import argparse

from tqdm import tqdm
from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='selfie2anime')
args = parser.parse_args()


transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_dateset = ImageDataset(f'../data/{args.dataset_name}', transform=transform, unaligned=False, mode='test')
val_dataloader = DataLoader(val_dateset, batch_size=1, shuffle=False, num_workers=0)

# load

USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

INPUT_SHAPE = (3, 256, 256)
N_RESIDUAL_BLOCK = 9

G_A2B = GenertorResNet(INPUT_SHAPE, N_RESIDUAL_BLOCK).to(DEVICE)
G_B2A = GenertorResNet(INPUT_SHAPE, N_RESIDUAL_BLOCK).to(DEVICE)

ckpt = torch.load(f'./saved_model/model_epoch_200.tar')
G_A2B.load_state_dict(ckpt['G_A2B'])
G_B2A.load_state_dict(ckpt['G_B2A'])


G_A2B.eval()
G_B2A.eval()

for idx, data in enumerate(tqdm(val_dataloader)):
  A = data['A'].to(DEVICE)
  B = data['B'].to(DEVICE)

  fake_B = G_A2B(A)
  fake_A = G_B2A(B)

  save_img(idx, f'./{args.dataset_name}_test_img', [A, B, fake_A, fake_B], 'test')