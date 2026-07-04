"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """Modular VGG block with configurable number of conv layers and channels.

    C configuration from Simonyan & Zisserman's VGG paper.
    """
    def __init__(self, in_channels, out_channels, num_convs, activation_class=nn.ReLU, padding=1):
        super().__init__()
        layers = []
        current_in_channels = in_channels
        for i in range(num_convs):
            is_config_c_tail = (num_convs == 3 and i == 2)
            kernel_size = 1 if is_config_c_tail else 3
            conv_padding = 0 if kernel_size == 1 else 1
            layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=kernel_size, padding=conv_padding))
            layers.append(nn.BatchNorm2d(out_channels))
            # Fix: was hardcoded nn.ReLU, silently ignoring config["ACTIVATION"].
            layers.append(activation_class(inplace=True))
            current_in_channels = out_channels # Fix: Update in_channels for next layer
            
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ResBlock with 3x3 convolutions (He et al., 2016)."""
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If spatial size shrinks (stride > 1) or channels change, adjust the shortcut
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        out = self.activation(out)
        return out


class AlexNet(nn.Module):
    """AlexNet (Krizhevsky et al., 2012) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        input_size = kwargs.get("input_size", 64) # Fix: Hardcoded input nn.linear of self.classifier
        # Fix: was hardcoded nn.ReLU below, silently ignoring config["ACTIVATION"].
        activation_str = kwargs.get("activation_str", "ReLU")
        activation_class = getattr(nn, activation_str)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(48),
            activation_class(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            activation_class(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_class(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_class(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            activation_class(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        flatten_dim = self._get_flatten_dim(in_channels, input_size)

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(flatten_dim, 1024),
            activation_class(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 1024),
            activation_class(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def _get_flatten_dim(self, in_channels, input_size):
        """Run a dummy tensor through self.features to infer the
        flattened feature size, so the classifier head is correct
        regardless of input resolution/channel count."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
        return out.flatten(1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG16(nn.Module):
    """VGG16 in C configuration of Simonyan & Zisserman, (2014) adapted for smaller inputs."""
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        input_size = kwargs.get("input_size", 64)
        # Fix: was hardcoded nn.ReLU below, silently ignoring config["ACTIVATION"].
        activation_str = kwargs.get("activation_str", "ReLU")
        activation_class = getattr(nn, activation_str)

        self.features = nn.Sequential(
            VGGBlock(in_channels, 64, num_convs=2, activation_class=activation_class),
            VGGBlock(64, 128, num_convs=2, activation_class=activation_class),
            VGGBlock(128, 256, num_convs=3, activation_class=activation_class),
            VGGBlock(256, 512, num_convs=3, activation_class=activation_class),
            VGGBlock(512, 512, num_convs=3, activation_class=activation_class)
        )

        # Fix: was a hardcoded nn.Linear(2048, ...), only correct for exactly
        # 64x64 inputs. Compute it dynamically instead, same as AlexNet.
        flatten_dim = self._get_flatten_dim(in_channels, input_size)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, 1024),
            activation_class(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(1024, 512),
            activation_class(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(512, num_classes)
        )

    def _get_flatten_dim(self, in_channels, input_size):
        """Run a dummy tensor through self.features to infer the
        flattened feature size, so the classifier head is correct
        regardless of input resolution/channel count."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.features(dummy)
        return out.flatten(1).shape[1]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet18(nn.Module):
    """ResNet18 (He et al., 2016) adapted for smaller inputs.
    
    activation - flexible activation function to allow experimentation (e.g., ReLU, LeakyReLU, etc.)
    """
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        activation_str = kwargs.get("activation_str", "ReLU")
        activation = getattr(nn, activation_str)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation(inplace=True)
        
        self.stage1 = nn.Sequential(
            ResBlock(64, 64, activation(inplace=True), stride=1),
            ResBlock(64, 64, activation(inplace=True), stride=1)
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128, activation(inplace=True), stride=2),          
            ResBlock(128, 128, activation(inplace=True), stride=1)
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256, activation(inplace=True), stride=2),
            ResBlock(256, 256, activation(inplace=True), stride=1)
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512, activation(inplace=True), stride=2),
            ResBlock(512, 512, activation(inplace=True), stride=1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class GreenVGG16(nn.Module):
    """
    GreenVGG16

    Lightweight VGG-style architecture designed for the Green Initiative.

    Compared to VGG16:
        - Reduced convolution channels
        - Smaller classifier
        - Lower parameter count
        - Lower memory footprint
        - Faster inference/training
    """

    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        drop_rate = kwargs.get("drop_rate", 0.5)
        input_size = kwargs.get("input_size", 64)

        activation_str = kwargs.get("activation_str", "ReLU")
        activation_class = getattr(nn, activation_str)

        # ---------------------------------------------------------
        # Feature extractor (compressed VGG16)
        # ---------------------------------------------------------

        self.features = nn.Sequential(

            VGGBlock(in_channels, 32, num_convs=2,
                     activation_class=activation_class),

            VGGBlock(32, 64, num_convs=2,
                     activation_class=activation_class),

            VGGBlock(64, 128, num_convs=3,
                     activation_class=activation_class),

            VGGBlock(128, 256, num_convs=3,
                     activation_class=activation_class),

            VGGBlock(256, 256, num_convs=3,
                     activation_class=activation_class),
        )

        flatten_dim = self._get_flatten_dim(
            in_channels,
            input_size
        )

        # ---------------------------------------------------------
        # Lightweight classifier
        # ---------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(flatten_dim, 512),

            activation_class(inplace=True),

            nn.Dropout(drop_rate),

            nn.Linear(512, 256),

            activation_class(inplace=True),

            nn.Dropout(drop_rate),

            nn.Linear(256, num_classes)
        )

    def _get_flatten_dim(self, in_channels, input_size):
        """
        Dynamically determine the flattened feature dimension.
        """

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                in_channels,
                input_size,
                input_size,
            )

            out = self.features(dummy)

        return out.flatten(1).shape[1]

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
