from cgi import test
from tabnanny import check
from utils import *
from dataset import *
from model import *

import numpy as np
import torch, os, imageio
from torchvision import transforms


USE_CUDA = True if torch.cuda.is_available() else False
DEVICE = 'cuda' if USE_CUDA else 'cpu'


transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## same img epoch 별로 test ###

test_path = '../data/facades/test/1.jpg'
test_img = Image.open(test_path)

w, h = test_img.size
real = test_img.crop((0, 0, w/2, h))
facade = test_img.crop((w/2, 0, w, h))

pix2pix_lst = os.listdir('./facades_checkpoint')


G = Pix2Pix_G().to(DEVICE)
G.eval()

n = 10
for name in pix2pix_lst:

  checkpoint = torch.load(f'facades_checkpoint/{name}')
  G.load_state_dict(checkpoint['G_net'])

  tensor = transform(facade).to(DEVICE).unsqueeze(0)
  g_img = G(tensor)

  x = np.asanyarray(real)
  fake = fn_tonumpy(fn_denorm(g_img, mean=0.5, std=0.5)).squeeze()
  y = np.asanyarray(facade)

  save_img(x, fake, y, f'./facades_img', n, test=True)

  n += 10


images = []
for f_name in os.listdir('./facades_img'):
  if 'fake' in f_name:
    images.append(imageio.v2.imread(os.path.join('./facades_img', f_name)))

imageio.mimsave('facade_test.gif', images, **{'duration': 1})



  

