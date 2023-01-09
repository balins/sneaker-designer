import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        def block(c_in, c_out, kernel_size, stride, padding, batch_norm=True, dropout=False):
            layers = [nn.Conv2d(c_in, c_out, kernel_size, stride,
                                padding, bias=False), nn.LeakyReLU(0.2, inplace=True)]

            if batch_norm:
                layers.insert(1, nn.BatchNorm2d(c_out))

            if dropout:
                layers.append(nn.Dropout2d(dropout))

            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            # input is (channels=3) x 64 x 64
            block(3, ndf, 4, 2, 1, batch_norm=False),

            # state size. (ndf) x 32 x 32
            block(ndf, ndf * 2, 4, 2, 1),

            # state size. (ndf*2) x 16 x 16
            block(ndf * 2, ndf * 4, 4, 2, 1, dropout=0.25),

            # state size. (ndf*4) x 8 x 8
            block(ndf * 4, ndf * 8, 4, 2, 1),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img)
