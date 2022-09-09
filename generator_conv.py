import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from torchsummary import summary
from torchvision import transforms

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



# EqualLinear
class EqualLinear(nn.Module):

    def __init__(
                self,
                in_features,
                out_features
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(in_features, out_features))

        self.bias = nn.Parameter(torch.zeros(out_features))

        self.scale = 1 / math.sqrt(in_features)

    
    def forward(self, x):
        out = x @ (self.weights * self.scale) + self.bias

        return out



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



class ContentExtractor(nn.ModuleList):

    def __init__(self):
        super().__init__(
            nn.ModuleList([
                conv_block(1, 16), # (16, 64, 64)
                conv_block(16, 32), # (32, 32, 32)
                conv_block(32, 64), # (64, 16, 16)
                conv_block(64, 128), # (128, 8, 8)
                conv_block(128, 256), # (256, 4, 4)
            ])
        )



class StyleExtractor(nn.Module):
    
    def __init__(
        self,
        style_n=16,
        class_n=108,
        style_dim=128
    ):

        super().__init__()

        self.style_n = style_n
        self.class_n = class_n
        self.style_dim = style_dim

        self.extractor = nn.ModuleList([])

        for _ in range(self.style_n):
            self.extractor.append(
                nn.Sequential(
                    EqualLinear(class_n, style_dim),
                    nn.LeakyReLU(0.2),

                    EqualLinear(style_dim, style_dim),
                    nn.LeakyReLU(0.2),

                    EqualLinear(style_dim, style_dim),
                    nn.LeakyReLU(0.2),

                    EqualLinear(style_dim, style_dim),
                    nn.LeakyReLU(0.2),
                )
            )

    

    def forward(self, x):
        # x shape: (batch_size, 1)

        x = F.one_hot(x.type(torch.long).squeeze(1), num_classes=self.class_n).type(torch.float32) # x shape: (batch_size, class_n)

        outs = []

        for i in range(self.style_n):
            outs.append(
                self.extractor[i](x).unsqueeze(2) # (batch_size, style_dim: 128, 1)
            )

        style = torch.cat(outs, dim=2) # (batch_size, style_dim, style_n)
        style = style.reshape(-1, self.style_dim, 4, 4)

        return style



class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.content_extractor = ContentExtractor()
        self.style_extractor = StyleExtractor()

        # input shape: (256+128, 4, 4)
        self.generator = nn.ModuleList([

            deconv_block(256+128, 256), # (256, 8, 8)

            deconv_block(256, 128), # (128, 16, 16)

            deconv_block(128, 64), # (64, 32, 32)

            deconv_block(64, 32), # (32, 64, 64)

            deconv_block(32, 16), # (16, 128, 128)

            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),

                WSConv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        ])


    def forward(self, content_letters, style_labels):
        self.content_list = [content_letters]

        for i in range(len(self.content_extractor)):
            self.content_list.append(self.content_extractor[i](self.content_list[-1]))
        
        self.style = self.style_extractor(style_labels)

        # (batch_size, 256+128, 4, 4)
        latent_vector = torch.cat((self.content_list[-1], self.style), dim=1)

        out = latent_vector

        for i in range(len(self.generator)):
            out = self.generator[i](out)

            if i < 3:
                out += F.interpolate(self.content_list[-i-1], scale_factor=2.0)

        return out


if __name__ == '__main__':

    generator = Generator()

    content_letters = torch.randn((2, 1, 128, 128))
    style_labels = torch.Tensor([[0], [1]])

    pred = generator(content_letters, style_labels)
    print(pred.shape)

    param_count = 0

    for p in generator.parameters():
        param_count += p.numel()

    print(param_count)
