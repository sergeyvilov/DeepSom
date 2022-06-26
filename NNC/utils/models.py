from torch import nn
import torch

class ConvNN(nn.Module):

    def __init__(self, dropout=0., target_width=150, target_height=70, kernel_size=3, n_channels=[32, 32, 32, 32, 256, 256, 128]):

        super().__init__()

        h, w = target_width, target_height

        self.conv1 = nn.Conv2d(14, n_channels[0], kernel_size=kernel_size, stride=1, padding=0)
        h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(n_channels[0], n_channels[1], kernel_size=kernel_size, stride=1, padding=0)
        h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer
        self.bn2 = nn.BatchNorm2d(n_channels[1])
        self.act2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        h, w = h//2, w//2 #size reduction in the MaxPool layer

        self.conv3 = nn.Conv2d(n_channels[1], n_channels[2], kernel_size=kernel_size, stride=1, padding=0)
        h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer
        self.bn3 = nn.BatchNorm2d(n_channels[2])
        self.act3 = nn.ReLU()
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        h, w = h//2, w//2 #size reduction in the MaxPool layer

        self.conv4 = nn.Conv2d(n_channels[2], n_channels[3], kernel_size=kernel_size, stride=1, padding=0)
        h, w = h-kernel_size+1, w-kernel_size+1 #size reduction in the Conv layer
        self.bn4 = nn.BatchNorm2d(n_channels[3])
        self.act4 = nn.ReLU()
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        h, w = h//2, w//2 #size reduction in the MaxPool layer

        self.flt = nn.Flatten()

        self.fc5 = nn.Linear(h * w * n_channels[3], n_channels[4])
        self.act5 = nn.ReLU()
        self.dp5 = nn.Dropout(dropout)

        self.fc6 = nn.Linear(n_channels[4], n_channels[5])
        self.act6 = nn.ReLU()
        self.dp6 = nn.Dropout(dropout)

        self.fc7 = nn.Linear(n_channels[5], n_channels[6])
        self.act7 = nn.ReLU()
        self.dp7 = nn.Dropout(dropout)

        self.fc8 = nn.Linear(n_channels[6], 1)
        self.act8 = nn.Sigmoid()


    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.mp2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.mp3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.mp4(out)

        out = self.flt(out)

        out = self.fc5(out)
        out = self.act5(out)
        out = self.dp5(out)

        out = self.fc6(out)
        out = self.act6(out)
        out = self.dp6(out)

        out = self.fc7(out)
        out = self.act7(out)
        out = self.dp7(out)

        out = self.fc8(out)
        out = self.act8(out)

        out = torch.squeeze(out,1)

        return out
