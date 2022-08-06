import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import os
import matplotlib.pyplot as plt

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

class CBR2d(nn.Module):
  def __init__(self,
    in_channel,
    out_channel,
    kernel_size = 4,
    stride = 2,
    padding = 1,
    bias = False,
    norm = True
  ):
    super().__init__()

    layer = [
      nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
      )
    ]

    if norm:
      layer.append(nn.BatchNorm2d(out_channel))
    
    layer.append(nn.LeakyReLU(0.2, inplace=True))

    self.enc = nn.Sequential(*layer)

  def forward(self, x):
    return self.enc(x)


class DECBR2d(nn.Module):
  def __init__(self,
    in_channel,
    out_channel,
    kernel_size = 4,
    stride = 2,
    padding = 1,
    bias = False,
    norm = True,
    drop = False
  ):
    super().__init__()

    self.norm = norm
    self.drop = drop

    self.upsample = nn.ConvTranspose2d(
      in_channels=in_channel,
      out_channels=out_channel,
      kernel_size=kernel_size,
      padding=padding,
      stride=stride,
      bias=bias
    ) 
    
    self.batchnorm = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU(inplace=True)
    
  
  def forward(self, x):
    x = self.upsample(x)
    
    if self.norm:
      x = self.batchnorm(x)
    
    if self.drop:
      x = F.dropout(x, 0.2, training=True)

    x = self.relu(x)

    return x 


def resize_and_randomcrop(data):
  x = data['real']
  y = data['facade']
  
  # B:0 C:1 H:2 W:3
  cat = torch.cat((x, y), dim=1) # channel 방향으로 합체

  transform = transforms.Compose([
    transforms.Resize((286, 286)),
    transforms.RandomCrop((256, 256))
  ])

  cat_trans= transform(cat) # resize and randomcrop

  x, y = torch.chunk(cat_trans, 2, dim=1) # 채널 방향으로 분리

  return x, y


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
      nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
      nn.init.normal_(m.weight.data, 1.0, 0.02)
      nn.init.constant_(m.bias.data, 0.0)


def save_net(G_net, D_net, G_optim, D_optim, ckpt_dir, epoch):
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  torch.save({
    'G_net': G_net.state_dict(), 
    'D_net': D_net.state_dict(), 
    'G_optim': G_optim.state_dict(),
    'D_optim': D_optim.state_dict()},
    f'{ckpt_dir}/model_epoch_{epoch}.tar') # 여러개 묶어서 저장하는 경우 tar파일이 일반적으로 사용된다.


def save_img(real, fake, input, dir, epoch, test=False):
  if not os.path.exists(dir):
    os.makedirs(dir)
  
  if test:
    plt.imsave(os.path.join(dir, f'test{epoch}_input.png'), input)
    plt.imsave(os.path.join(dir, f'test{epoch}_real.png'), real)
    plt.imsave(os.path.join(dir, f'test{epoch}_fake.png'), fake)

  else:
    plt.imsave(os.path.join(dir, f'{epoch}_input.png'), input)
    plt.imsave(os.path.join(dir, f'{epoch}_real.png'), real)
    plt.imsave(os.path.join(dir, f'{epoch}_fake.png'), fake)



def show_train_hist(hist, show = False):
  x = range(len(hist['D_loss_lst']))

  y1 = hist['D_loss_lst']
  y2 = hist['G_loss_lst']

  plt.plot(x, y1, label='D_loss')
  plt.plot(x, y2, label='G_loss')

  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.legend()
  plt.grid(True)
  plt.tight_layout()

  if show:
    plt.show()
  else:
    plt.close()