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

        Returns
        -------
        x + residual : tensor
            Sum of tensors on the residual path and the straight path.
        """

        residual = self.res(x)
        x = self.bn1(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)

        return x + residual

class DownBlock(nn.Module):
    """
    Down Block Class
    """
    def __init__(self, input_channels, output_channels):
        """
        Constructor to build a DownBlock
        Uses 2 Residual blocks, One AvgPooling2D layer

        Args
        ----
        input_channels : int
                        Number of input channels
        output_channels : int
                        Number of output channels
        """
        super().__init__()
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.res1 = ResidualBlock(self.in_ch, self.out_ch)
        self.res2 = ResidualBlock(self.out_ch, self.out_ch)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        """
        Forward pass for the DownBlock

        Args
        ----
        x : tensor
            Input tensor to down block

        Returns
        -------
        x : tensor
            Output tensor from down block
        
        skips : list of tensors
            Skip connections to feed to UpBlocks
        """
        skips = []
        x = self.res1(x)
        skips.append(x)
        x = self.res2(x)
        skips.append(x)
        x = self.avg_pool(x)

        return x, skips

class UpBlock(nn.Module):
    """
    Up Block Class
    """
    def __init__(self, input_channels, output_channels, skip_channels):
        """
        Constructor for Up Block.
        Uses one Upsampling2D with bilinear interpolation and two Residual Blocks

        Args
        ----
        input_channels : int
                        Number of input channels
        output_channels : int
                        Number of output channels
        skip_channels : int
                        Number of channels from skip connection
        """
        super().__init__()
        self.in_ch = input_channels
        self.out_ch = output_channels
        self.skip_ch = skip_channels
        self.ups = nn.UpsamplingBilinear2d(scale_factor=2)
        self.res1 = ResidualBlock(self.in_ch + self.skip_ch, self.out_ch)
        self.res2 = ResidualBlock(self.out_ch + self.skip_ch, self.out_ch)

    def forward(self, x, skips):
        """
        Forward pass for UpBlock

        Args
        ----
        x : tensor
            Input tensor to UpBlock
        
        skips : list of tensors
            Skip connections list

        Returns
        -------
        x : tensor
            Output tensor from UpBlock
        """
        x = self.ups(x)
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.res2(x)

        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x_T, noise_level):
        pass

    def sinusoidal_embedding(self, noise):
        pass
