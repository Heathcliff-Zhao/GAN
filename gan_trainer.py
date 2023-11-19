"""implementation of the trainer class"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os


class Trainer():
    def __init__(self, generator, discriminator, trainloader, testloader, device, epochs, lr, z_dim, sample_size, sample_path, model_path, conditional=False):
        self.generator = generator
        self.discriminator = discriminator
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.sample_path = sample_path + ('_conditional' if conditional else '_unconditional')
        self.model_path = model_path + ('_conditional' if conditional else '_unconditional')
        self.conditional = conditional
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.criterion = nn.BCELoss()
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr)

    def train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.epochs):
            for i, (real_images, real_number) in enumerate(self.trainloader):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                if not self.conditional:
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                else:
                    real_number = real_number.to(self.device)
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)
                # train discriminator
                if not self.conditional:
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_images = self.generator(z)
                else:
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_images = self.generator(z, real_number)
                # fake_images = self.generator(z)
                if not self.conditional:
                    d_real_loss = self.criterion(self.discriminator(real_images), real_labels)
                    d_fake_loss = self.criterion(self.discriminator(fake_images), fake_labels)
                else:
                    d_real_loss = self.criterion(self.discriminator(real_images, real_number), real_labels)
                    d_fake_loss = self.criterion(self.discriminator(fake_images, real_number), fake_labels)
                d_loss = d_real_loss + d_fake_loss
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                # train generator
                if not self.conditional:
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_images = self.generator(z)
                else:
                    z = torch.randn(batch_size, self.z_dim).to(self.device)
                    fake_images = self.generator(z, real_number)
                # fake_images = self.generator(z)
                if not self.conditional:
                    g_loss = self.criterion(self.discriminator(fake_images), real_labels)
                else:
                    g_loss = self.criterion(self.discriminator(fake_images, real_number), real_labels)
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                if (i + 1) % 100 == 0:
                    dg_z = self.discriminator(fake_images).mean().item() if not self.conditional else self.discriminator(fake_images, real_number).mean().item()
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'.format(epoch, self.epochs, i + 1, len(self.trainloader), d_loss.item(), g_loss.item(), real_labels.mean().item(), dg_z))

    def save_image(self, epoch):
        z = torch.randn(self.sample_size, self.z_dim).to(self.device)
        if not self.conditional:
            fake_images = self.generator(z)
        else:
            labels = torch.randint(0, 10, (self.sample_size,)).to(self.device)
            fake_images = self.generator(z, labels)
        # fake_images = self.generator(z)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        save_image(fake_images, self.sample_path + '/fake_images-{}.png'.format(epoch), nrow=8)

    def save_model(self):
        torch.save(self.generator.state_dict(), self.model_path + '/generator.pth')
        torch.save(self.discriminator.state_dict(), self.model_path + '/discriminator.pth')
