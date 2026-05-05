import torch
import torch.nn as nn
import math

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


class UNet(nn.Module):
    """
    UNet Class
    """
    def __init__(self, input_channels, output_channels):
        """
        Constructor for the UNet Class.
        Implements the initial convoltuion and concatenation.
        Passes through 3 downblocks.
        Passes through 2 Residualblocks.
        Passes through 3 Upblocks.
        Final Conv layer to predict the noise.

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
        self.noise_embedding_size = 32
        
        #First convolution
        self.conv = nn.Conv2d(self.in_ch, 32, kernel_size=3, stride=1, padding=1)
        
        #Upsampling
        self.ups_noise = nn.UpsamplingNearest2d(size=128)
        
        #Down Blocks
        self.d1 = DownBlock(64,32)
        self.d2 = DownBlock(32,64)
        self.d3 = DownBlock(64,96)

        #Residual Blocks
        self.r1 = ResidualBlock(96,128)
        self.r2 = ResidualBlock(128,128)

        #Up Blocks
        self.u1 = UpBlock(128,96,96)
        self.u2 = UpBlock(96,64,64)
        self.u3 = UpBlock(64,32,32)
        self.final_conv = nn.Conv2d(32, self.out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x_T, noise_var):
        """
        Forward pass for U-Net

        Args
        ----
        x_T : tensor
            Input noise
        
        noise_var : tensor
            Noice variance for a batch

        Returns
        -------
        noise : tensor
            Predicted noise
        """

        all_skips = []
        
        x = self.conv(x_T)
        embedding_noise = self._sinusoidal_embedding(noise_var)
        embedding_noise = self.ups_noise(embedding_noise)

        x = torch.cat([x,embedding_noise], dim=1)
        
        x , skips = self.d1(x)
        all_skips.extend(skips)
        x , skips = self.d2(x)
        all_skips.extend(skips)
        x, skips = self.d3(x)
        all_skips.extend(skips)
        
        x = self.r1(x)
        x = self.r2(x)

        x = self.u1(x, all_skips)
        x = self.u2(x, all_skips)
        x = self.u3(x, all_skips)

        x = self.final_conv(x)

        return x


    def _sinusoidal_embedding(self, noise_var):
        """
        Sinusoidal Embedding for the noise variance

        Args
        ----
        noise_var : tensor
                Noise variance for a batch
        """
        if noise_var.dim() == 1:
            noise_var = noise_var[:, None]

        frequencies = torch.exp(
            torch.linspace(math.log(1.0),
                           math.log(1000.0), 
                           self.noise_embedding_size //2,
                           device=noise_var.device,
                           dtype=noise_var.dtype)
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = torch.cat(
            [
                torch.sin(angular_speeds * noise_var), 
                torch.cos(angular_speeds * noise_var)
            ],
            dim=1
        )
        return embeddings[: , : , None, None]

if __name__ == '__main__':
    unet_model = UNet(4,3)
    test_tensor = torch.randn(1,4,128,128)
    test_noise = torch.tensor([[0.4]])

    result = unet_model(test_tensor, test_noise)
    print('Size of the input tensor is ', test_tensor.shape)
    print('Size of the output tensor is ', result.shape)
