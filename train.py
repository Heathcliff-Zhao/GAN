from dataset import MNIST_dataset
from model import Generator, Discriminator, ConditionalGenerator, ConditionalDiscriminator
from gan_trainer import Trainer
import torch
import argparse

args = argparse.ArgumentParser()
args.add_argument('--conditional', action='store_true')
args = args.parse_args()

input_size = 784
output_size = 784
hidden_dim = 32
z_dim = 100
batch_size = 128
num_classes = 10
embedding_dim = 32
epochs = 50
lr = 0.0002
sample_size = 64
sample_path = './samples'
model_path = './models'
conditional = args.conditional

# load dataset
dataset = MNIST_dataset('./data', batch_size)
trainloader, testloader = dataset.get_dataset()
# create model
if not conditional:
    generator = Generator(z_dim, output_size, hidden_dim)
    discriminator = Discriminator(input_size, hidden_dim)
else:
    generator = ConditionalGenerator(z_dim, output_size, hidden_dim, num_classes, embedding_dim)
    discriminator = ConditionalDiscriminator(input_size, hidden_dim, num_classes, embedding_dim)
# create trainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(generator, discriminator, trainloader, testloader, device, epochs, lr, z_dim, sample_size, sample_path, model_path, conditional=conditional)
# train model
trainer.train()