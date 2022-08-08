from decimal import Decimal
from DL.pix2pix.dataset import CityscapesDataset, Edges2shoesDataset, MapsDataset
from dataset import FacadesDataset
from model import *
from utils import *

import argparse
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

data_dirs = {
  'edges2shoes': '../data/edges2shoes/train',
  'facades': '../data/facades/train',
  'maps': '../data/maps/train',
  'cityscapes': '../data/cityscapes/train'
}



USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='facades')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--L1_lambda', type=float, default=100.0)
args = parser.parse_args()

# DataLoader

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = data_dirs[args.dataset]

if args.dataset == 'facades':
  data_train = FacadesDataset(data_dir=data_dir, transform=transform)
  loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
elif args.dataset == 'edges2shoes':
  data_train = Edges2shoesDataset(data_dir=data_dir, transform=transform)
  loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
elif args.dataset == 'maps':
  data_train = MapsDataset(data_dir=data_dir, transform=transform)
  loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
elif args.dataset == 'cityscapes':
  data_train = CityscapesDataset(data_dir=data_dir, transform=transform)
  loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)


G = Pix2Pix_G().to(DEVICE)
D = Pix2Pix_D().to(DEVICE)

# init weight
G.apply(weights_init_normal)
D.apply(weights_init_normal)

# train mode
G.train()
D.train()

# Loss
BCE_loss = nn.BCELoss().to(DEVICE)
L1_loss = nn.L1Loss().to(DEVICE)

# optimizer
G_optim = optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
D_optim = optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

G_loss_lst = []
D_loss_lst = []


print('TRAIN START')
G.train()
D.train()
for epoch in range(1, args.n_epoch+1):

  for data in loader_train:

    # img resize and crop -> gpu 로 옮기고? 아니면 그전에?
    real, input_data = resize_and_randomcrop(data) # jittering

    if USE_CUDA:
      real = real.cuda()
      input_data = input_data.cuda()
    
    fake_img = G(input_data)

    # train D
    D.zero_grad()

    D_output = D(torch.cat((real, input_data), dim=1)).squeeze() # 30 30 1 -> 30 30
    D_real_loss = BCE_loss(D_output, torch.ones(D_output.size()).to(DEVICE))

    D_output = D(torch.cat((fake_img.detach(), input_data), dim=1)).squeeze()
    D_fake_loss = BCE_loss(D_output, torch.zeros(D_output.size()).to(DEVICE))

    D_loss = (D_real_loss + D_fake_loss) / 2 # -> /2는 굳이 안해도 된다
    D_loss.backward()

    D_optim.step()


    # train G
    G.zero_grad()

    D_output = D(torch.cat((fake_img, input_data), dim=1)).squeeze()
    G_BCE = BCE_loss(D_output, torch.ones(D_output.size()).to(DEVICE))
    G_L1 =  L1_loss(fake_img, real)
    G_loss = G_BCE + (args.L1_lambda * G_L1)
    G_loss.backward()
    
    G_optim.step()

    
  D_loss_lst.append(D_loss.item())
  G_loss_lst.append(G_loss.item())

  
  print(f'{epoch:03d}/{args.n_epoch:03d}\t Loss_G: {G_loss.item():.3f}(BCE: {G_BCE:.3f}, L1: {G_L1:.3f})\t Loss_D: {D_loss.item():.3f}')

  if epoch % 10 == 0:
    save_net(G, D, G_optim, D_optim,f'./{args.dataset}_checkpoint', epoch)
    
    real = fn_tonumpy(fn_denorm(real, mean=0.5, std=0.5)).squeeze()
    fake = fn_tonumpy(fn_denorm(fake_img, mean=0.5, std=0.5)).squeeze()
    input_data = fn_tonumpy(fn_denorm(input_data, mean=0.5, std=0.5)).squeeze()

    save_img(real, fake, input_data, f'./{args.dataset}_img', epoch)


hist = {
  'D_loss_lst': D_loss_lst,
  'G_loss_lst': G_loss_lst
}

show_train_hist(hist, show=True)

    












