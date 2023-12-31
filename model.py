import torch
import torch.nn as nn

# 生成器网络
class Generator(nn.Module):
    def __init__(self, conditional=False):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入是 Z, 进入卷积
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False) if not conditional else nn.ConvTranspose2d(200, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 状态大小. (512) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 状态大小. (256) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 状态大小. (128) x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 状态大小. (64) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 最终状态大小. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    

# 条件生成器网络
class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(10, 100)
        # self.projector = nn.Linear(640, 100)
        self.model = Generator(conditional=True)  # 与前面的生成器相同
        # self.factor = nn.Parameter(torch.ones(1))

    def forward(self, noise, labels):
        # print(noise.shape, labels.shape)
        # 将标签转换为one-hot向量
        c = self.label_embedding(labels)
        # print(c.shape)
        # c = c.view(c.size(0), -1)
        # c = self.projector(c)
        c = c.view(c.size(0), 100, 1, 1)
        # x = self.factor * c + noise
        x = torch.cat([c, noise], dim=1)
        # print(x.shape)
        return self.model(x)
    

    

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, conditional=False):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入是 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False) if not conditional else nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小. (64) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小. (128) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小. (256) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态大小. (512) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

    

# 条件判别器网络
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 64 * 64 * 3)
        # self.projector = nn.Linear(640, 64 * 64 * 3)
        self.model = Discriminator(conditional=True)  # 与前面的判别器相同
        # trainable factor
        self.factor = nn.Parameter(torch.ones(1))

    def forward(self, img, labels):
        # 将标签转换为one-hot向量
        c = self.label_embedding(labels)
        # print(c.shape)
        # print(img.shape)
        # print(labels.shape)
        # c = c.view(c.size(0), -1)
        # c = self.projector(c)
        c = c.view(c.size(0), 3, 64, 64)
        # x = self.factor * c + img
        # print(c.shape, img.shape)
        x = torch.cat([img, c], 1)
        return self.model(x)


