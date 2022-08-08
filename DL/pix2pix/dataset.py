import os
from PIL import Image
from torch.utils.data import Dataset


class FacadesDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.data_lst = os.listdir(self.data_dir)
    self.transform = transform

  def __len__(self):
    return len(self.data_lst)

  def __getitem__(self, idx):

    img = Image.open(os.path.join(self.data_dir, self.data_lst[idx]))
    w, h = img.size
    real = img.crop((0, 0, w/2, h))
    facade = img.crop((w/2, 0, w, h))

    if self.transform:
      real = self.transform(real)
      facade = self.transform(facade)

    data = {'real': real, 'input': facade}
    
    return data


class Edges2shoesDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.data_lst = os.listdir(self.data_dir)
    self.transform = transform

  def __len__(self):
    return len(self.data_lst)

  def __getitem__(self, idx):

    img = Image.open(os.path.join(self.data_dir, self.data_lst[idx]))
    w, h = img.size
    edge = img.crop((0, 0, w/2, h))
    real = img.crop((w/2, 0, w, h))

    if self.transform:
      real = self.transform(real)
      edge = self.transform(edge)

    data = {'real': real, 'input': edge}
    
    return data


class CityscapesDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.data_lst = os.listdir(self.data_dir)
    self.transform = transform

  def __len__(self):
    return len(self.data_lst)

  def __getitem__(self, idx):

    img = Image.open(os.path.join(self.data_dir, self.data_lst[idx]))
    w, h = img.size
    real = img.crop((0, 0, w/2, h))
    semantic = img.crop((w/2, 0, w, h))

    if self.transform:
      real = self.transform(real)
      semantic = self.transform(semantic)

    data = {'real': real, 'input': semantic}
    
    return data


class MapsDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.data_lst = os.listdir(self.data_dir)
    self.transform = transform

  def __len__(self):
    return len(self.data_lst)

  def __getitem__(self, idx):

    img = Image.open(os.path.join(self.data_dir, self.data_lst[idx]))
    w, h = img.size
    real = img.crop((0, 0, w/2, h))
    line = img.crop((w/2, 0, w, h))

    if self.transform:
      real = self.transform(real)
      line = self.transform(line)

    data = {'real': real, 'input': line}
    
    return data