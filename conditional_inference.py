from model import ConditionalGenerator
import torch
from torchvision.utils import save_image

model_path = './models_conditional'
z_dim = 100
output_size = 784
hidden_dim = 32


# create model
generator = ConditionalGenerator(z_dim, output_size, hidden_dim, 10, 32)
# pth
generator.load_state_dict(torch.load(model_path + '/generator.pth'))
generator.eval()

def generate_image(label):
    z = torch.randn(1, z_dim)
    label = torch.tensor([label])
    fake_image = generator(z, label)
    fake_image = fake_image.view(fake_image.size(0), 1, 28, 28)
    save_image(fake_image, f'try_fake_image{label.item()}.png')


if __name__ == '__main__':
    while True:
        keyin = input('Enter a label: ')
        if keyin == 'q':
            break
        else:
            label = int(keyin)
        generate_image(label)
