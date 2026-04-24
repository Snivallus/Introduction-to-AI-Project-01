import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LeNet(nn.Module):
    """LeNet-style CNN for CIFAR-10 image classification.

    Original architecture adapted for 32x32 RGB images.
    Consists of two convolutional layers and three fully connected layers.
    """

    def __init__(self, dropout: Optional[float] = None) -> None:
        super(LeNet, self).__init__()

        # Convolutional layer: 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully connected layers: y = Wx + b
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        # Optional dropout layer for regularization
        self.dropout = None
        if dropout is not None:
            if isinstance(dropout, float) and 0 < dropout < 1:
                self.dropout = nn.Dropout(dropout)
            else:
                raise ValueError("dropout must be a float between 0 and 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Output tensor of shape (batch_size, 10).
        """
        # [batch_size, 3, 32, 32] -> conv1 -> [batch_size, 6, 28, 28] -> relu -> maxpool -> [batch_size, 6, 14, 14]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # [batch_size, 6, 14, 14] -> conv2 -> [batch_size, 16, 10, 10] -> relu -> maxpool -> [batch_size, 16, 5, 5]
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten: [batch_size, 16 * 5 * 5] = [batch_size, 400]
        x = x.view(x.size()[0], -1)
        # [batch_size, 400] -> fc1 -> [batch_size, 120]
        x = F.relu(self.fc1(x))
        # [batch_size, 120] -> dropout (optional) -> [batch_size, 120]
        if self.dropout is not None:
            x = self.dropout(x)
        # [batch_size, 120] -> fc2 -> [batch_size, 84]
        x = F.relu(self.fc2(x))
        # [batch_size, 84] -> fc3 -> [batch_size, 10]
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block for constructing deeper CNNs.

    Consists of two 3x3 convolutional layers with batch normalization
    and a skip connection (identity shortcut).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 3x3 convolution: (batch, in_channels, H, W) -> (batch, out_channels, H/stride, W/stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 convolution: (batch, out_channels, H/stride, W/stride) -> same shape (stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            Output tensor of shape (batch_size, out_channels, H/stride, W/stride).
        """
        identity = x
        # [batch, C_in, H, W] -> conv1 -> [batch, C_out, H/s, W/s] -> bn1 -> relu -> [batch, C_out, H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        # [batch, C_out, H/s, W/s] -> conv2 -> [batch, C_out, H/s, W/s] -> bn2 -> [batch, C_out, H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        # Downsample the identity if needed to match output dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        # Skip connection: element-wise addition (shape unchanged)
        out += identity
        out = F.relu(out)
        return out


class MyCNN(nn.Module):
    """Modern CNN (ResNet-style) for CIFAR-10 image classification.

    Implements a deeper architecture with residual blocks, batch normalization,
    and adaptive average pooling. Designed for 32x32 RGB input images.
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        # Initial convolution: [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in_channels = 64
        # Residual stages, each halving spatial size when stride=2:
        #   layer1: [batch, 64, 32, 32] -> [batch, 64, 32, 32]   (stride=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        #   layer2: [batch, 64, 32, 32] -> [batch, 128, 16, 16]  (stride=2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        #   layer3: [batch, 128, 16, 16] -> [batch, 256, 8, 8]   (stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        #   layer4: [batch, 256, 8, 8] -> [batch, 512, 4, 4]     (stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        # Global pooling: [batch, 512, 4, 4] -> [batch, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        # Classifier: [batch, 512] -> [batch, num_classes]
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # [batch, 3, 32, 32] -> conv1 -> [batch, 64, 32, 32] -> bn1/relu -> [batch, 64, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # [batch, 64, 32, 32] -> layer1 -> [batch, 64, 32, 32]
        x = self.layer1(x)
        # [batch, 64, 32, 32] -> layer2 -> [batch, 128, 16, 16]
        x = self.layer2(x)
        # [batch, 128, 16, 16] -> layer3 -> [batch, 256, 8, 8]
        x = self.layer3(x)
        # [batch, 256, 8, 8] -> layer4 -> [batch, 512, 4, 4]
        x = self.layer4(x)
        # [batch, 512, 4, 4] -> avgpool -> [batch, 512, 1, 1]
        x = self.avgpool(x)
        # [batch, 512, 1, 1] -> flatten -> [batch, 512]
        x = torch.flatten(x, 1)
        # [batch, 512] -> dropout -> [batch, 512]
        x = self.dropout(x)
        # [batch, 512] -> fc -> [batch, num_classes]
        x = self.fc(x)
        return x