from torch import nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: 100 x 1
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #  1024 x 4x4
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 512 x 8x8
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 256 x 16x16
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 128 x 32x32
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64 x 64x64
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(4, 3),
                stride=(4, 3),
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # 32 x 256x192
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=nc,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Tanh(),
            # nc x 256x192
        )

    def forward(self, input):
        return self.main(input)
