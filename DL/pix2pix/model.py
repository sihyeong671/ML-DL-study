from matplotlib import transforms
import torch
import torch.nn as nn

from utils import *

class Pix2Pix_G(nn.Module):
  def __init__(self):
    super().__init__()

    # (B 256 256 64)
    self.down1 = CBR2d(in_channel=3, out_channel=64, norm=False) # (B 128 128 64)
    self.down2 = CBR2d(in_channel=64, out_channel=128) # (B 64 64 128)
    self.down3 = CBR2d(in_channel=128, out_channel=256) # (B 32 32 256)
    self.down4 = CBR2d(in_channel=256, out_channel=512) # (B 16 16 512)
    self.down5 = CBR2d(in_channel=512, out_channel=512) # (B 8 8 512)
    self.down6 = CBR2d(in_channel=512, out_channel=512) # (B 4 4 512)
    self.down7 = CBR2d(in_channel=512, out_channel=512) # (B 2 2 512)
    self.down8 = CBR2d(in_channel=512, out_channel=512, norm=False) # (B 1 1 512) # batch_size가 1이면 norm적용 불가능

    self.up8 = DECBR2d(in_channel=512, out_channel=512, drop=True) # (B 2 2 512)
    self.up7 = DECBR2d(in_channel=1024, out_channel=512, drop=True) # (B 4 4 512) 
    self.up6 = DECBR2d(in_channel=1024, out_channel=512, drop=True) # (B 8 8 512)
    self.up5 = DECBR2d(in_channel=1024, out_channel=512) # (B 16 16 512)
    self.up4 = DECBR2d(in_channel=1024, out_channel=256) # (B 32 32 512)
    self.up3 = DECBR2d(in_channel=512, out_channel=128) # (B 64 64 128)
    self.up2 = DECBR2d(in_channel=256, out_channel=64) # (B 128 128 64)
    self.up1 = nn.ConvTranspose2d(
      in_channels=128,
      out_channels=3, 
      kernel_size=4, 
      padding=1, 
      stride=2
    ) # (B 256 256 3)


  def forward(self, x):
    down1 = self.down1(x)
    down2 = self.down2(down1)
    down3 = self.down3(down2)
    down4 = self.down4(down3)
    down5 = self.down5(down4)
    down6 = self.down6(down5)
    down7 = self.down7(down6)
    down8 = self.down8(down7)

    up8 = self.up8(down8)
    cat7 = torch.cat((up8, down7), dim=1)
    up7 = self.up7(cat7)
    cat6 = torch.cat((up7, down6), dim=1)
    up6 = self.up6(cat6)
    cat5 = torch.cat((up6, down5), dim=1)
    up5 = self.up5(cat5)
    cat4 = torch.cat((up5, down4), dim=1)
    up4 = self.up4(cat4)
    cat3 = torch.cat((up4, down3), dim=1)
    up3 = self.up3(cat3)
    cat2 = torch.cat((up3, down2), dim=1)
    up2 = self.up2(cat2)
    cat1 = torch.cat((up2, down1), dim=1)
    up1 = self.up1(cat1)

    return torch.tanh(up1)


class Pix2Pix_D(nn.Module):
  def __init__(self):
    super().__init__()

    # (B 256 256 3)
    self.enc1 = CBR2d(in_channel=6, out_channel=64, norm=False)
    self.enc2 = CBR2d(in_channel=64, out_channel=128)
    self.enc3 = CBR2d(in_channel=128, out_channel=256)
    self.enc4 = CBR2d(in_channel=256, out_channel=512, stride=1, padding=0)
    self.enc5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)
    
  def forward(self, x):
    # x 는 img, label 합친 tensor
    x = self.enc1(x)
    x = self.enc2(x)
    x = self.enc3(x)
    x = self.enc4(x)
    x = self.enc5(x)
    return torch.sigmoid(x)
    














  

  