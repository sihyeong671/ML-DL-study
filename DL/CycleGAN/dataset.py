import glob
import os
from PIL import Image
import random
from utils import to_rgb

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
  def __init__(self, root, transform=None, unaligned=False, mode='train'):
    self.transform= transform
    self.unaligned=unaligned

    if mode == 'train':
      self.file_A = sorted(glob.glob(os.path.join(root, 'trainA')+'/*.*'))
      self.file_B = sorted(glob.glob(os.path.join(root, 'trainB')+'/*.*'))
    else:
      self.file_A = sorted(glob.glob(os.path.join(root, 'testA')+'/*.*'))
      self.file_B = sorted(glob.glob(os.path.join(root, 'testB')+'/*.*'))

  def __len__(self):
    return max(len(self.file_A), len(self.file_B))

  def __getitem__(self, index):
    img_A = Image.open(self.file_A[index % len(self.file_A)])
    
    if self.unaligned:
      img_B = Image.open(self.file_B[random.randint(0, len(self.file_B)-1)])
    else:
      img_B = Image.open(self.file_B[index % len(self.file_B)])

    if img_A.mode != 'RGB':
      img_A = to_rgb(img_A)
    
    if img_B.mode != 'RGB':
      img_B = to_rgb(img_B)

    if self.transform:
      item_A = self.transform(img_A)
      item_B = self.transform(img_B)

    return {'A': item_A, 'B': item_B}
  
  



