"""
This code is adjusted from the repository: https://github.com/gegewen/ufno
Modifications include adapting the 3D architecture to 2D for image processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 2D Spectral Convolution Layer
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# ==========================================
# 2. Custom 2D U-Net
# ==========================================
class U_net2d(nn.Module):
    def __init__(self, input_channels, output_channels, dropout_rate=0.0):
        super(U_net2d, self).__init__()
        self.input_channels = input_channels

        # Encoder
        self.conv1 = self.conv(input_channels, output_channels, stride=2, dropout_rate=dropout_rate)
        self.conv2 = self.conv(output_channels, output_channels, stride=2, dropout_rate=dropout_rate)
        self.conv2_1 = self.conv(output_channels, output_channels, stride=1, dropout_rate=dropout_rate)
        self.conv3 = self.conv(output_channels, output_channels, stride=2, dropout_rate=dropout_rate)
        self.conv3_1 = self.conv(output_channels, output_channels, stride=1, dropout_rate=dropout_rate)

        # Decoder
        self.deconv2 = self.deconv(output_channels, output_channels)
        self.deconv1 = self.deconv(output_channels * 2, output_channels)
        self.deconv0 = self.deconv(output_channels * 2, output_channels)

        self.output_layer = self.output(output_channels * 2, output_channels, stride=1)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv2 = self.deconv2(out_conv3)
        out_deconv2 = self.match_size(out_deconv2, out_conv2)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        out_deconv1 = self.deconv1(concat2)
        out_deconv1 = self.match_size(out_deconv1, out_conv1)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)

        out_deconv0 = self.deconv0(concat1)
        out_deconv0 = self.match_size(out_deconv0, x)
        concat0 = torch.cat((x, out_deconv0), 1)

        out = self.output_layer(concat0)
        return out

    def match_size(self, x, target):
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x

    def conv(self, in_planes, output_channels, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_planes, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, stride):
        return nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)

# ==========================================
# 3. Core Block Definitions
# ==========================================
class FNOBlock2d(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(FNOBlock2d, self).__init__()
        self.conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.w = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x_res = x
        x_fourier = self.conv(x)
        x_local = self.w(x)
        x = F.gelu(x_fourier + x_local) + x_res
        return x

class UFNOBlock2d(nn.Module):
    def __init__(self, channels, modes1, modes2):
        super(UFNOBlock2d, self).__init__()
        self.conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.w = nn.Conv2d(channels, channels, 1)
        self.unet = U_net2d(channels, channels)

    def forward(self, x):
        x_res = x
        x_fourier = self.conv(x)
        x_local = self.w(x)
        x_unet = self.unet(x)
        x = F.gelu(x_fourier + x_local + x_unet) + x_res
        return x

# ==========================================
# 4. Main Network Architecture
# ==========================================
class ufno(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, modes1=32, modes2=32, width=64, num_fno_layers=2, num_ufno_layers=2):
        super(ufno, self).__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lifting = nn.Linear(self.in_channels, self.width)

        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(channels=self.width, modes1=modes1, modes2=modes2)
            for _ in range(num_fno_layers)
        ])

        self.ufno_blocks = nn.ModuleList([
            UFNOBlock2d(channels=self.width, modes1=modes1, modes2=modes2)
            for _ in range(num_ufno_layers)
        ])

        self.projection1 = nn.Linear(self.width, 128)
        self.projection2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)

        for block in self.fno_blocks:
            x = block(x)

        for block in self.ufno_blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)

        x = F.gelu(self.projection1(x))
        output = self.projection2(x)

        output = output.permute(0, 3, 1, 2)
        return output
