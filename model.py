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



# Sequential 안에서 바로 Reshape 해줌.
class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()

        self.shape = shape

    
    def forward(self, x):
        return x.view(-1, *self.shape)



class Upsampling(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=2.0)


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



class WSLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        std = 1 / math.sqrt(in_features)

        nn.init.normal_(self.linear.weight, mean=0.0, std=std)
        nn.init.zeros_(self.linear.bias)

    
    def forward(self, x):
        return self.linear(x)



class AdaIN(nn.Module):

    def __init__(self, style_dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale_factor=1.0):
        super().__init__()

        self.eps = 1e-5

        self.affine = WSLinear(style_dim, in_channels*2)
        self.channels = in_channels

        self.scale_factor = scale_factor

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leakyrelu = nn.LeakyReLU(0.2)
    

    def forward(self, x, style):
        # x shape: (batch_size, channels, H, W)
        # style shape: (batch_size, style_dim)

        x = F.interpolate(x, scale_factor=self.scale_factor)

        style = self.affine(style) # (batch_size, channels*2)
        style = style.view(-1, self.channels*2, 1, 1)
        
        x_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        x_std = torch.std(x, dim=(2, 3), keepdim=True)

        out = (x - x_mean) / (x_std + self.eps)
        out = out * style[:, :self.channels, :, :] + style[:, self.channels:, :, :]
        
        out = self.conv(out)
        out = self.leakyrelu(out)

        return out


class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()

        # input shape: (1, 128, 128)
        self.first_content_extractor = nn.Sequential(
            nn.Sequential(
                WSConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (16, 128, 128)
            
            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (64, 32, 32)

            nn.Sequential(
                WSConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (128, 16, 16)

            nn.Sequential(
                WSConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (256, 8, 8)

            nn.Sequential(
                WSConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (512, 4, 4)
        )


        # input shape: (1, 128, 128)
        self.middle_content_extractor = nn.Sequential(
            nn.Sequential(
                WSConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (16, 128, 128)
            
            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (64, 32, 32)

            nn.Sequential(
                WSConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (128, 16, 16)

            nn.Sequential(
                WSConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (256, 8, 8)

            nn.Sequential(
                WSConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (512, 4, 4)
        )


        # input shape: (1, 128, 128)
        self.last_content_extractor = nn.Sequential(
            nn.Sequential(
                WSConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (16, 128, 128)
            
            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (64, 32, 32)

            nn.Sequential(
                WSConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (128, 16, 16)

            nn.Sequential(
                WSConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (256, 8, 8)

            nn.Sequential(
                WSConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2)
            ), # (512, 4, 4)
        )


        # input shape: (1, 128, 128)
        self.font_style_extractor = nn.Sequential(
            nn.Sequential(
                WSConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            ), # shape: (16, 64, 64)

            nn.Sequential(
                WSConv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            ), # shape: (64, 32, 32)

            nn.Sequential(
                WSConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            ), # shape: (128, 16, 16)

            nn.Sequential(
                WSConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            ), # shape: (256, 8, 8)
            
            nn.Sequential(
                WSConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(2, 2),
            ), # shape: (512, 4, 4)

            nn.Sequential(
                nn.Flatten(),
                WSLinear(in_features=512*4*4, out_features=512*6),
                nn.LeakyReLU(0.2)
            ),
        
            Reshape(shape=(6, 512))
        )

        adain_channel_pairs = [(512, 256), (256, 128), (128, 64)]
        
        # input shape: (512, 4, 4)
        self.first_generator = nn.ModuleList([])

        for in_channels, out_channels in adain_channel_pairs:
            self.first_generator.append(AdaIN(style_dim=512, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.first_generator.append(AdaIN(style_dim=512, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2.0))


        self.middle_generator = nn.ModuleList([])

        for in_channels, out_channels in adain_channel_pairs:
            self.middle_generator.append(AdaIN(style_dim=512, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.middle_generator.append(AdaIN(style_dim=512, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2.0))


        self.last_generator = nn.ModuleList([])

        for in_channels, out_channels in adain_channel_pairs:
            self.last_generator.append(AdaIN(style_dim=512, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
            self.last_generator.append(AdaIN(style_dim=512, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, scale_factor=2.0))

        
        # input shape: (64*3, 32, 32)
        self.merger = nn.Sequential(
            WSConv2d(in_channels=64*3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            Upsampling(),

            WSConv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            Upsampling(),

            WSConv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2)
        )


    def forward(self, input, first, middle, last):
        first_content = self.first_content_extractor(first)
        middle_content = self.middle_content_extractor(middle)
        last_content = self.last_content_extractor(last)

        styles = self.font_style_extractor(input)

        first_out = first_content
        middle_out = middle_content
        last_out = last_content

        for i in range(6):
            first_out = self.first_generator[i](first_out, style=styles[:, i, :])
            middle_out = self.middle_generator[i](middle_out, style=styles[:, i, :])
            last_out = self.last_generator[i](last_out, style=styles[:, i, :])

        # (batch_size, channels_n, H, W)
        letter_latent_vector = torch.cat((first_out, middle_out, last_out), dim=1)

        out = self.merger(letter_latent_vector)

        return out


class Loss(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device)
        self.feature_extractor.eval()


    def forward(self, pred, label):
        pred_features = self.feature_extractor(pred.repeat(1, 3, 1, 1))
        label_features = self.feature_extractor(label.repeat(1, 3, 1, 1))

        vgg_loss = F.mse_loss(pred_features, label_features)
        mse_loss = F.mse_loss(pred, label)

        return vgg_loss, mse_loss

    
if __name__ == '__main__':
    device = torch.device('cpu')

    generator = Generator().to(device)
    
    input = torch.randn((2, 1, 128, 128)).to(device).type(torch.float32)
    first = torch.randn((2, 1, 128, 128)).to(device).type(torch.float32)
    middle = torch.randn((2, 1, 128, 128)).to(device).type(torch.float32)
    last = torch.randn((2, 1, 128, 128)).to(device).type(torch.float32)
    output = torch.randn((2, 1, 128, 128)).to(device).type(torch.float32)

    criterion = Loss(device=device)

    start_time = time()
    pred = generator(input, first, middle, last)
    print(pred.shape)

    vgg_loss, mse_loss = criterion(pred, output)

    total_loss = vgg_loss + mse_loss

    print(f"total loss: {total_loss}, vgg_loss: {vgg_loss}, mse_loss: {mse_loss}, time: {time() - start_time}")

    total_loss.backward()

    for p in generator.font_style_extractor.parameters():
        print(p.shape, p.grad[0][0][0][0])
        break