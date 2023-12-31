from dataset import get_dataloader
from model import Generator, Discriminator, ConditionalGenerator, ConditionalDiscriminator
from gan_trainer import Trainer
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument('--conditional', action='store_true')
args = args.parse_args()

epochs = 50
lr = 0.0002
sample_path = './samples2'
model_path = './models'
conditional = args.conditional
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
trainloader = get_dataloader()
# create model
if not conditional:
    generator = Generator()
    discriminator = Discriminator()
else:
    generator = ConditionalGenerator()
    # discriminator = ConditionalDiscriminator(input_size, hidden_dim, num_classes, embedding_dim)
    discriminator = ConditionalDiscriminator()
# create trainer
trainer = Trainer(generator.to(device), discriminator.to(device), trainloader, device, epochs, lr, sample_path, model_path, conditional)
# train model
trainer.train()