"""build gan model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc3(x), 0.2))
        x = torch.tanh(self.fc4(x))
        return x
    

class ConditionalGenerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_classes, embed_dim):
        super(ConditionalGenerator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        # assert input_size == embed_dim + input_size
        self.fc1 = nn.Linear(input_size + embed_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, y):
        y = self.embed(y)
        x = torch.cat([x, y], dim=1)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc3(x), 0.2))
        x = torch.tanh(self.fc4(x))
        return x
    

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = torch.sigmoid(self.fc3(x))
        return x
    

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, embed_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        # assert input_size == embed_dim + input_size
        self.fc1 = nn.Linear(input_size + embed_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = self.embed(y)
        x = torch.cat([x, y], dim=1)
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.2))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.2))
        x = torch.sigmoid(self.fc3(x))
        return x


if __name__ == '__main__':
    # test generator
    input_size = 784
    output_size = 784
    hidden_dim = 32
    z_dim = 100
    z = torch.randn(2, z_dim)
    generator = Generator(z_dim, output_size, hidden_dim)
    print(generator)
    output = generator(z)
    print(output.shape)
    # test discriminator
    input_size = 784
    hidden_dim = 32
    x = torch.randn(2, 784)
    discriminator = Discriminator(input_size, hidden_dim)
    print(discriminator)
    output = discriminator(x)
    print(output.shape)