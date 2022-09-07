import random
import torch
from torchvision.utils import save_image, make_grid 
from PIL import Image
import os


def to_rgb(img): # gray to rgb
  rgb_img = Image.new('RGB', img.size)
  rgb_img.paste(img)
  return rgb_img


def weight_init(m): # 가중치 초기화
  classname = m.__class__.__name__

  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0.0)

  elif classname.find('BatchNorm2d') != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)


# 스케줄러
class LambdaLR:
  def __init__(self, n_epochs, offset, decay_start_epoch):
    assert(n_epochs - decay_start_epoch) > 0
    "Decay must start befor the training session ends!"
    self.n_epochs = n_epochs
    self.offset = offset
    self.decay_start_epoch = decay_start_epoch
  
  def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def save_img(epoch, dir, imgs, mode):
  if not os.path.exists(dir):
    os.makedirs(dir)
  
  real_A = imgs[0]
  real_B = imgs[1]
  fake_B = imgs[2]
  fake_A = imgs[3]
  if mode == 'train':
    real_A = make_grid(real_A, nrow=1, normalize=True)
    real_B = make_grid(real_B, nrow=1, normalize=True)
    fake_A = make_grid(fake_A, nrow=1, normalize=True)
    fake_B = make_grid(fake_B, nrow=1, normalize=True)
  else:
    real_A = make_grid(real_A, nrow=1, normalize=True)
    real_B = make_grid(real_B, nrow=1, normalize=True)
    fake_A = make_grid(fake_A, nrow=1, normalize=True)
    fake_B = make_grid(fake_B, nrow=1, normalize=True)


  img_grid = torch.cat((real_A, fake_A, real_B, fake_B), 1)
  save_image(img_grid, f'{dir}/{epoch}.png', normalize=False)


class ReplayBuffer:
  def __init__(self, max_size=50):
    assert max_size > 0, 'Empty buffer or trying to create a black hole'

    self.max_size = max_size
    self.data = []

  def push_and_pop(self, data):
    to_return = []

    for element in data.data:
      element = torch.unsqueeze(element, 0)
      if len(self.data) < self.max_size:
        self.data.append(element)
        to_return.append(element)
      else:
        if random.uniform(0, 1) > 0.5:
          i = random.randint(0, self.max_size-1)
          to_return.append(self.data[i].clone())
        else:
          to_return.append(element)
        
    return torch.cat(to_return)


def save_net(models, ckpt_dir, epoch):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  torch.save({
    'G_A2B': models['G_A2B'].state_dict(), 
    'G_B2A': models['G_B2A'].state_dict()},
    f'./{ckpt_dir}/model_epoch_{epoch}.tar')
