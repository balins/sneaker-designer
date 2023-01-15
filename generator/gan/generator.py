import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super().__init__()

        def block(c_in, c_out, kernel_size, stride, padding, dropout=False):
            layers = [
                nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.1, inplace=True)
            ]

            layers.append(nn.Dropout2d(dropout))

            return nn.Sequential(*layers)


        self.net = nn.Sequential(
            # input is Z, going into a convolution
            block(nz, ngf * 8, 4, 1, 0),

            # state size. (ngf*8) x 4 x 4
            block(ngf * 8, ngf * 4, 4, 2, 1),
            
            # state size. (ngf*4) x 8 x 8
            block(ngf * 4, ngf * 2, 4, 2, 1, dropout=0.2),

            # state size. (ngf*2) x 16 x 16
            block(ngf * 2, ngf, 4, 2, 1),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            # state size. (channels=3) x 64 x 64
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
