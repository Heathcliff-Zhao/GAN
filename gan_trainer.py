import torch
import torch.nn as nn
import torch.optim as optim
from utils import save_image, weights_init
import os
import torchvision.utils as vutils
import wandb


class Trainer():
    def __init__(self, generator, discriminator, trainloader, device, epochs, lr, sample_path, model_path, conditional=False):
        self.generator = generator
        self.discriminator = discriminator
        self.trainloader = trainloader
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.sample_path = sample_path + ('_conditional' if conditional else '_unconditional')
        self.model_path = model_path + ('_conditional' if conditional else '_unconditional')
        self.project_name = 'conditional_gan' if conditional else 'unconditional_gan'
        self.conditional = conditional
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.criterion = nn.BCELoss()
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(64, 100, 1, 1, device=device)
        self.fixed_labels = torch.randint(0, 10, (64,), device=device)
        # self.real_label = 1
        # self.fake_label = 0
        self.real_labels = torch.ones(self.trainloader.batch_size, device=self.device)
        self.fake_labels = torch.zeros(self.trainloader.batch_size, device=self.device)
        # self.num_epochs = 50
        self.img_list = []
        self.init_wandb()

    def init_wandb(self):
        wandb.init(project='GAN', name=self.project_name)

    def train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainloader, 0):
                # 训练判别器: 最大化 log(D(x)) + log(1 - D(G(z)))
                ## 更新判别器网络: maximize log(D(x)) + log(1 - D(G(z)))
                self.discriminator.zero_grad()
                # 训练真实图像
                real_cpu = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_cpu.size(0)
                # label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
                
                output = self.discriminator(real_cpu) if not self.conditional else self.discriminator(real_cpu, labels)
                errD_real = self.criterion(output, self.real_labels)
                errD_real.backward()
                D_x = output.mean().item()

                # 训练假图像
                noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
                fake = self.generator(noise) if not self.conditional else self.generator(noise, labels)
                # label.fill_(self.fake_label)
                output = self.discriminator(fake.detach()) if not self.conditional else self.discriminator(fake.detach(), labels)
                errD_fake = self.criterion(output, self.fake_labels)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.d_optimizer.step()

                # 更新生成器网络: maximize log(D(G(z)))
                self.generator.zero_grad()
                # label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.discriminator(fake) if not self.conditional else self.discriminator(fake, labels)
                errG = self.criterion(output, self.real_labels)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.g_optimizer.step()

                print(f'[{epoch}/{self.epochs}][{i}/{len(self.trainloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')
                wandb.log({'epoch': epoch, 'loss_d': errD.item(), 'loss_g': errG.item(), 'd_x': D_x, 'd_g_z1': D_G_z1, 'd_g_z2': D_G_z2})

            # 每个epoch结束时保存生成器的输出
            with torch.no_grad():
                fake = self.generator(self.fixed_noise).detach().cpu() if not self.conditional else self.generator(self.fixed_noise, self.fixed_labels).detach().cpu()
            self.img_list.append(vutils.make_grid(fake, padding=5, normalize=True))
        
        self.save_model()
        save_image(real_cpu, self.img_list, self.sample_path, self.device)

    def save_model(self):
        torch.save(self.generator.state_dict(), self.model_path + '/generator.pth')
        torch.save(self.discriminator.state_dict(), self.model_path + '/discriminator.pth')
