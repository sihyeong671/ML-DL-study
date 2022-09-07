import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import itertools

from tqdm import tqdm
import time
import random
import argparse
import os

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='monet2photo')
parser.add_argument('--n_residual_blocks', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lambda_cyc', type=float, default=10.0)
parser.add_argument('--lambda_id', type=float, default=5.0)
parser.add_argument('--seed', default=99)
parser.add_argument('--ckpt_interval', type=int, default=10)
args = parser.parse_args()


# 초기화

INPUT_SHAPE = (3, 256, 256)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'


#
criterion_gan = nn.MSELoss().to(DEVICE)
criterion_cycle = nn.L1Loss().to(DEVICE)
criterion_identity = nn.L1Loss().to(DEVICE)

G_A2B = GenertorResNet(INPUT_SHAPE, args.n_residual_blocks).to(DEVICE)
G_B2A = GenertorResNet(INPUT_SHAPE, args.n_residual_blocks).to(DEVICE)
D_A = Discriminator(INPUT_SHAPE).to(DEVICE)
D_B = Discriminator(INPUT_SHAPE).to(DEVICE)

G_A2B.apply(weight_init)
G_B2A.apply(weight_init)
D_A.apply(weight_init)
D_B.apply(weight_init)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# 
optimizer_G = optim.Adam(
  itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
  lr=args.lr,
  betas=(args.b1, args.b2)
)

optimizer_D = optim.Adam(
  itertools.chain(D_A.parameters(), D_B.parameters()),
  lr=args.lr,
  betas=(args.b1, args.b2)
)


# 스케줄러
lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(args.n_epochs, 0, 100).step)
lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(args.n_epochs, 0, 100).step)


transform = transforms.Compose([
  transforms.Resize(286, Image.Resampling.BICUBIC),
  transforms.RandomCrop((256, 256)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageDataset(f'../data/{args.dataset_name}', transform, unaligned=True)
dataloader = DataLoader(
  dataset,
  batch_size=args.batch_size, 
  shuffle=True,
  num_workers=0
)

print('TRAIN START')
for epoch in range(1, args.n_epochs+1):
  for idx, batch in enumerate(tqdm(dataloader)):
    
    real_A = batch['A'].to(DEVICE)
    real_B = batch['B'].to(DEVICE)

    valid = torch.ones((real_A.size(0), *D_A.output_shape)).to(DEVICE)
    fake = torch.zeros((real_A.size(0), *D_A.output_shape)).to(DEVICE)

    # train G
    G_A2B.train()
    G_B2A.train()

    optimizer_G.zero_grad()

    loss_id_A = criterion_identity(G_B2A(real_A), real_A)
    loss_id_B = criterion_identity(G_A2B(real_B), real_B)
    loss_identity = loss_id_A + loss_id_B

    # gan loss
    fake_A = G_B2A(real_B)
    loss_G_B2A = criterion_gan(D_A(fake_A), valid)
    fake_B = G_A2B(real_A)
    loss_G_A2B = criterion_gan(D_B(fake_B), valid)
    loss_gan = loss_G_A2B + loss_G_B2A
    
    # cycle loss
    recov_A = G_B2A(fake_B)
    loss_cycle_A = criterion_cycle(recov_A, real_A)
    recov_B = G_A2B(fake_A)
    loss_cycle_B = criterion_cycle(recov_B, real_B)
    loss_cycle = loss_cycle_A + loss_cycle_B

    loss_G = loss_gan * args.lambda_cyc + loss_cycle * args.lambda_id

    loss_G.backward()
    optimizer_G.step()

    # train D
    optimizer_D.zero_grad()

    loss_real_B = criterion_gan(D_B(real_B), valid)
    fake_B_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake_B = criterion_gan(D_B(fake_B_.detach()), fake)

    loss_real_A = criterion_gan(D_A(real_A), valid)
    fake_A_ = fake_B_buffer.push_and_pop(fake_B)
    loss_fake_A = criterion_gan(D_A(fake_A_.detach()), fake)

    loss_D = loss_real_A + loss_fake_A + loss_real_B + loss_fake_B
    loss_D.backward()
    optimizer_D.step()

    
  save_img(epoch, f'./{args.dataset_name}_train_img',[real_A, real_B, fake_A, fake_B], 'train')
  print(f'[Epoch {epoch}/{args.n_epochs}] [D loss: {loss_D.item():.3f}] [G loss: {loss_G.item():.3f}(adv:{loss_gan:.3f}, cycle:{loss_cycle:.3f}, identity:{loss_identity:.3f})]')

  lr_scheduler_D.step()
  lr_scheduler_G.step()

  if epoch % args.ckpt_interval == 0:
    models = {'G_A2B': G_A2B,'G_B2A': G_B2A}
    save_net(models, 'monet_saved_model', epoch)

print('TRAIN FIN')
