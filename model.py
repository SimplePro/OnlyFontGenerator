import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights

from PIL import Image

from os import listdir

from tqdm import tqdm

import math

from time import time



class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()

        self.shape = shape

    
    def forward(self, x):
        return x.view(-1, *self.shape)


class WSConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, transposed=False):
        super().__init__()

        if transposed: self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else: self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        
        if type(kernel_size) == int:
            std = 1 / math.sqrt(in_channels * kernel_size**2)
        
        elif type(kernel_size) == tuple:
            std = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])

        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()

        def conv_block(in_channels, out_channels, downsampling=True):
            return nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels),

                WSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels),

                nn.MaxPool2d(2, 2) if downsampling else nn.Sequential()
            )

        def deconv_block(in_channels, out_channels):
            return nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, transposed=True),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels),

                WSConv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, transposed=True),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_channels),
            )

        # input shape: (1, 128, 128)
        self.content_extractor = nn.ModuleList([
            conv_block(1, 16), # (16, 64, 64)
            conv_block(16, 64), # (64, 32, 32)
            conv_block(64, 128), # (128, 16, 16)
            conv_block(128, 256), # (256, 8, 8)
            conv_block(256, 512), # (512, 4, 4)
        ])

        # input shape: (1, 128, 128)
        self.style_extractor = nn.Sequential(
            conv_block(1, 4), # (16, 64, 64)
            conv_block(4, 16), # (64, 32, 32)
            conv_block(16, 32), # (32, 16, 16)
            conv_block(32, 64), # (64, 8, 8)
            conv_block(64, 128), # (128, 4, 4)
        )

        # input shape: (512+128, 4, 4)
        self.generator = nn.ModuleList([
            deconv_block(512+128, 512), # (512, 8, 8)
            deconv_block(512, 256), # (256, 16, 16)
            deconv_block(256, 128), # (128, 32, 32)
            deconv_block(128, 64), # (64, 64, 64)
            deconv_block(64, 16), # (16, 128, 128)
            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),

                WSConv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        ])


    def forward(self, input, gothic):
        self.content_list = [gothic]

        for i in range(len(self.content_extractor)):
            self.content_list.append(self.content_extractor[i](self.content_list[-1]))
        
        self.style = self.style_extractor(input)

        # (batch_size, 512+128, 4, 4)
        latent_vector = torch.cat((self.content_list[-1], self.style), dim=1)

        out = latent_vector

        for i in range(len(self.generator)):
            out = self.generator[i](out)

            if i < 3:
                out += F.interpolate(self.content_list[-i-1], scale_factor=2.0)

        return out


class DiscriminatorConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # (C_step, H, W) -> (C_step, H, W)
        self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

        # (C_step, H, W) -> (C_(step-1), H, W)
        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        # (C_(step-1), H/2, W/2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.leakyrelu = nn.LeakyReLU(0.2)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.batchnorm1(out)

        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.batchnorm2(out)

        out = self.downsample(out)

        return out

    
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        # input shape: (1, 128, 128)
        self.main = nn.Sequential(
            DiscriminatorConvBlock(1, 16), # (16, 64, 64)
            DiscriminatorConvBlock(16, 64), # (64, 32, 32)
            DiscriminatorConvBlock(64, 128), # (128, 16, 16)
            DiscriminatorConvBlock(128, 256), # (256, 8, 8)
            DiscriminatorConvBlock(256, 512), # (512, 4, 4)

            WSConv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),

            WSConv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),

            nn.Flatten()
        )

        self.style_classifier = nn.Sequential(
            DiscriminatorConvBlock(1, 16), # (16, 64, 64)
            DiscriminatorConvBlock(16, 64), # (64, 32, 32)
            DiscriminatorConvBlock(64, 128), # (128, 16, 16)
            DiscriminatorConvBlock(128, 256), # (256, 8, 8)
            DiscriminatorConvBlock(256, 512), # (512, 4, 4)

            WSConv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),

            WSConv2d(in_channels=512, out_channels=108, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        return self.main(x), self.style_classifier(x)