import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block Class
    """
    def __init__(self, input_channels, output_channels):
        """
        Constructor to build a Residual Block.
        Uses BatchNorm, 2 Convolution layers.
        Activation is SiLU(same as Swish in TF)

        Args
        ----
        input_channels : int
                        Number of input channels
        output_channels : int
                        Number of output channels
        """
        super().__init__()
        self.in_chan = input_channels
        self.out_chan = output_channels
        self.bn1 = nn.BatchNorm2d(self.in_chan)
        self.conv1 = nn.Conv2d(self.in_chan, self.out_chan, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_chan, self.out_chan, kernel_size=3, padding=1)
        if(self.in_chan == self.out_chan):
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(self.in_chan, self.out_chan, kernel_size=1)

        self.activation = nn.SiLU()
    
    def forward(self, x):
        """
        Forward pass of the Residual block.

        Args
        ----
        x : tensor
            input tensor
        """

        residual = self.res(x)
        x = self.bn1(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)

        return x + residual

class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

class UpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x_T, noise_level):
        pass

    def sinusoidal_embedding(self, noise):
        pass
